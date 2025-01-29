use crate::buffer;
use crate::str;
use std::io::{BufRead, Read};

#[derive(Clone, Copy, Debug)]
pub enum TokenKind {
    BlockComment,
    LineComment,
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
    OpenBrace,
    CloseBrace,
    NumericLiteral(usize),
    StringLiteral(str::Id),
    Ident(str::Id),
    Fn,
    Impl,
    Where,
    Let,
    Const,
    For,
    If,
    Else,
    FatArrow,
    LArrow,
    Lt,
    Gt,
    Eq,
    EqEq,
    And,
    AndAnd,
    Or,
    OrOr,
    Plus,
    Minus,
    Slash,
    Star,
    Percent,
    Pound,
    Colon,
    Semi,
    Point,
    Comma,
}

impl TokenKind {
    fn render<'a>(self, interner: &'a str::Interner) -> &'a str {
        match self {
            TokenKind::BlockComment => "block comment",
            TokenKind::LineComment => "line comment",
            TokenKind::OpenParen => "(",
            TokenKind::CloseParen => ")",
            TokenKind::OpenBracket => "[",
            TokenKind::CloseBracket => "]",
            TokenKind::OpenBrace => "{",
            TokenKind::CloseBrace => "}",
            TokenKind::NumericLiteral(_) => "numeric literal",
            TokenKind::StringLiteral(id) => interner.lookup(id),
            TokenKind::Ident(id) => interner.lookup(id),
            TokenKind::Fn => "fn",
            TokenKind::Impl => "impl",
            TokenKind::Where => "where",
            TokenKind::Let => "let",
            TokenKind::Const => "const",
            TokenKind::For => "for",
            TokenKind::If => "if",
            TokenKind::Else => "else",
            TokenKind::FatArrow => "=>",
            TokenKind::LArrow => "->",
            TokenKind::Lt => "<",
            TokenKind::Gt => ">",
            TokenKind::Eq => "=",
            TokenKind::EqEq => "≡",
            TokenKind::And => "&",
            TokenKind::AndAnd => "∧",
            TokenKind::Or => "|",
            TokenKind::OrOr => "∨",
            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Star => "*",
            TokenKind::Slash => "/",
            TokenKind::Percent => "%",
            TokenKind::Pound => "#",
            TokenKind::Colon => ":",
            TokenKind::Semi => ";",
            TokenKind::Point => ".",
            TokenKind::Comma => ",",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Location {
    pub row: u32,
    pub col: u32,
}

impl Location {
    fn new() -> Self {
        Self { row: 1, col: 1 }
    }
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.row, self.col)
    }
}

impl Location {
    fn consume(&mut self, bytes: &[u8]) {
        *self = bytes
            .utf8_chunks()
            .fold(*self, |loc: Location, chunk: std::str::Utf8Chunk<'_>| {
                loc.consume_chunk(chunk)
            });
    }

    fn consume_chunk(self, chunk: std::str::Utf8Chunk<'_>) -> Location {
        chunk
            .valid()
            .chars()
            .fold(self, |loc: Location, char: char| loc.consume_char(char))
    }

    fn consume_char(self, char: char) -> Location {
        match char {
            '\n' => Self {
                row: self.row + 1,
                col: 1,
            },
            _ => Self {
                col: self.col + 1,
                row: self.row,
            },
        }
    }
}

pub struct PeekReader<R: ?Sized> {
    lookahead: usize,
    location: Location,
    buf: buffer::Buffer,
    inner: R,
}

impl<R: Read> PeekReader<R>
where
    R: Read,
{
    pub fn with_capacity(capacity: usize, inner: R) -> Self {
        Self {
            lookahead: 0usize,
            location: Location::new(),
            inner: inner,
            buf: buffer::Buffer::with_capacity(capacity),
        }
    }
}

impl<R: Read> PeekAhead for PeekReader<R> {
    type Item = u8;
    type Location = Location;

    fn peek(&self) -> &[Self::Item] {
        &self.buf.buffer()[..self.lookahead()]
    }

    fn lookahead(&self) -> usize {
        self.lookahead
    }

    fn location(&self) -> Self::Location {
        self.location
    }

    fn consume(&mut self, amt: usize) {
        self.location.consume(&self.buf.buffer()[..amt]);
        self.buf.consume(amt);
        self.lookahead = self.lookahead.saturating_sub(amt);
    }

    fn advance(&mut self, amt: usize) -> Result<(), PeekError> {
        if self.buf.pos() + self.lookahead() + amt < self.buf.filled() {
            self.lookahead += amt;
            return Ok(());
        }

        if self.lookahead() + amt >= self.buf.capacity() {
            return Err(PeekError::TooLong);
        }

        self.buf.backshift();
        self.buf.read_more(&mut self.inner)?;
        if self.lookahead + amt > self.buf.filled() - self.buf.pos() {
            self.lookahead = self.buf.filled() - self.buf.pos();
            return Err(PeekError::Eof);
        };

        self.lookahead += amt;
        Ok(())
    }
}

pub struct Lexer<'a, 'b, R> {
    inner: &'a mut PeekReader<R>,
    interner: &'b mut str::Interner,
}

impl<'a, 'b, R: Read> Lexer<'a, 'b, R> {
    pub fn new(inner: &'a mut PeekReader<R>, interner: &'b mut str::Interner) -> Self {
        Self {
            inner: inner,
            interner: interner,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Token {
    location: Location,
    kind: TokenKind,
}

impl<'a, 'b, 'c, R: Read> IntoIterator for &'c mut Lexer<'a, 'b, R> {
    type Item = Token;
    type IntoIter = Tokens<'a, 'b, 'c, R>;

    fn into_iter(self) -> Self::IntoIter {
        Tokens::new(self)
    }
}

pub struct Tokens<'a, 'b, 'c, R> {
    inner: &'c mut Lexer<'a, 'b, R>,
}

impl<'a, 'b, 'c, R: Read> Tokens<'a, 'b, 'c, R> {
    fn new(inner: &'c mut Lexer<'a, 'b, R>) -> Self {
        Self { inner: inner }
    }

    fn emit(&mut self, kind: TokenKind) -> TokenKind {
        self.inner.inner.consume(1);
        kind
    }

    fn identifier(&mut self) -> TokenKind {
        self.inner
            .inner
            .take_while(|x| is_identifier(x))
            .consume_with(|x| match x {
                b"fn" => TokenKind::Fn,
                b"impl" => TokenKind::Impl,
                b"where" => TokenKind::Where,
                b"let" => TokenKind::Let,
                b"const" => TokenKind::Const,
                b"for" => TokenKind::For,
                b"if" => TokenKind::If,
                b"else" => TokenKind::Else,
                _ => TokenKind::Ident(self.inner.interner.intern_bytes(x)),
            })
    }

    fn number(&mut self) -> Option<TokenKind> {
        self.inner
            .inner
            .take_while(|x| is_numeric(x))
            .consume_with(|x| {
                std::str::from_utf8(x)
                    .ok()
                    .and_then(|x| x.parse::<usize>().ok())
                    .map(TokenKind::NumericLiteral)
            })
    }

    fn string(&mut self) -> TokenKind {
        // Skip leading '"'
        self.inner.inner.consume(1);
        let literal = self
            .inner
            .inner
            .take_while(|x| x != &b'"')
            .consume_with(|x| TokenKind::StringLiteral(self.inner.interner.intern_bytes(x)));
        // Skip trailing '"'
        self.inner.inner.consume(1);
        literal
    }

    fn slash(&mut self) -> Option<TokenKind> {
        self.inner.inner.consume(1);
        Some(match self.inner.inner.peek_last().ok()? {
            b'/' => {
                self.inner.inner.skip_while(|x| x != &b'\n');
                self.inner.inner.consume(1);
                TokenKind::LineComment
            }
            // b'*' => {
            //     self.inner.consume(1);
            //     TokenKind::EqEq
            // }
            _ => TokenKind::Slash,
        })
    }

    fn minus(&mut self) -> Option<TokenKind> {
        self.inner.inner.consume(1);
        Some(match self.inner.inner.peek_last().ok()? {
            b'>' => {
                self.inner.inner.consume(1);
                TokenKind::LArrow
            }
            // b'*' => {
            //     self.inner.consume(1);
            //     TokenKind::EqEq
            // }
            _ => TokenKind::Minus,
        })
    }

    fn eqeq(&mut self) -> Option<TokenKind> {
        self.inner.inner.consume(1);
        Some(match self.inner.inner.peek_last().ok()? {
            b'=' => {
                self.inner.inner.consume(1);
                TokenKind::EqEq
            }
            b'>' => {
                self.inner.inner.consume(1);
                TokenKind::FatArrow
            }
            _ => TokenKind::Eq,
        })
    }

    fn andand(&mut self) -> Option<TokenKind> {
        self.inner.inner.consume(1);
        Some(match self.inner.inner.peek_last().ok()? {
            b'&' => {
                self.inner.inner.consume(1);
                TokenKind::AndAnd
            }
            _ => TokenKind::And,
        })
    }

    fn oror(&self) -> Option<TokenKind> {
        todo!()
    }
}

fn is_whitespace(byte: &u8) -> bool {
    matches!(*byte, b'\t' | b'\n' | b'\x0C' | b'\r' | b' ')
}

fn is_identifier(byte: &u8) -> bool {
    matches!(*byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
}

fn is_numeric(byte: &u8) -> bool {
    matches!(*byte, b'0'..=b'9')
}

impl<'a, 'b, 'c, R> Iterator for Tokens<'a, 'b, 'c, R>
where
    R: Read,
{
    type Item = Token;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.inner.skip_while(|x| is_whitespace(x));
        let location = self.inner.inner.location();
        let kind = match self.inner.inner.peek_last().ok()? {
            b'/' => self.slash()?,
            b'=' => self.eqeq()?,
            b'&' => self.andand()?,
            b'|' => self.oror()?,
            b'-' => self.minus()?,
            b'*' => self.emit(TokenKind::Star),
            b'%' => self.emit(TokenKind::Percent),
            b'+' => self.emit(TokenKind::Plus),
            b':' => self.emit(TokenKind::Colon),
            b';' => self.emit(TokenKind::Semi),
            b'#' => self.emit(TokenKind::Pound),
            b'.' => self.emit(TokenKind::Point),
            b',' => self.emit(TokenKind::Comma),
            b'(' => self.emit(TokenKind::OpenParen),
            b')' => self.emit(TokenKind::CloseParen),
            b'[' => self.emit(TokenKind::OpenBracket),
            b']' => self.emit(TokenKind::CloseBracket),
            b'{' => self.emit(TokenKind::OpenBrace),
            b'}' => self.emit(TokenKind::CloseBrace),
            b'<' => self.emit(TokenKind::Lt),
            b'>' => self.emit(TokenKind::Gt),
            b'a'..=b'z' | b'A'..=b'Z' => self.identifier(),
            b'0'..=b'9' => self.number()?,
            b'"' => self.string(),
            _ => return None,
        };
        Some(Token { location, kind })
    }
}

#[derive(Debug)]
pub enum PeekError {
    TooLong,
    Eof,
    IoError(std::io::Error),
}

impl From<std::io::Error> for PeekError {
    fn from(error: std::io::Error) -> Self {
        PeekError::IoError(error)
    }
}

pub trait PeekAhead: Sized {
    type Item;
    type Location;

    fn peek(&self) -> &[Self::Item];

    fn advance(&mut self, amt: usize) -> Result<(), PeekError>;

    fn consume(&mut self, amt: usize);

    fn location(&self) -> Self::Location;

    fn lookahead(&self) -> usize {
        self.peek().len()
    }

    fn take_while<F>(&mut self, f: F) -> TakeWhile<'_, Self, F>
    where
        F: FnMut(&Self::Item) -> bool,
    {
        TakeWhile::new(self, f)
    }

    fn skip_while<F>(&mut self, f: F) -> usize
    where
        F: FnMut(&Self::Item) -> bool,
    {
        self.take_while(f).consume_with(|y| y.len())
    }

    fn peek_next(&mut self, amt: usize) -> Result<&[Self::Item], PeekError> {
        if self.lookahead() < amt {
            self.advance(amt - self.lookahead())?;
        }
        Ok(self.peek())
    }

    fn peek_last(&mut self) -> Result<&Self::Item, PeekError> {
        self.peek_next(1)
            .and_then(|x| x.last().ok_or(PeekError::Eof))
    }
}

#[derive(Debug)]
pub struct TakeWhile<'a, T, F> {
    peekable: &'a mut T,
    take_while: F,
    taken: usize,
    done: bool,
}

impl<'a, T, F> TakeWhile<'a, T, F>
where
    T: PeekAhead,
    F: FnMut(&T::Item) -> bool,
{
    pub fn new(peekable: &'a mut T, take_while: F) -> Self {
        Self {
            taken: 0,
            peekable,
            take_while,
            done: false,
        }
    }

    pub fn consume_with<G, U>(mut self, mut g: G) -> U
    where
        G: FnMut(&[T::Item]) -> U,
    {
        while let Some(_) = self.next() {}
        let head = &self.peekable.peek()[..self.taken];
        let x = g(head);
        self.peekable.consume(head.len());
        x
    }
}

impl<'a, T, F> Iterator for TakeWhile<'a, T, F>
where
    T: PeekAhead,
    F: FnMut(&T::Item) -> bool,
{
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        match self
            .peekable
            .peek_next(self.taken + 1)
            .ok()
            .and_then(|x| x.last())
        {
            Some(token) if (self.take_while)(token) => {
                self.taken += 1;
                Some(())
            }
            _ => {
                self.done = true;
                None
            }
        }
    }
}

// [             capacity              ]
// [ consumed |       remaining        ]
// [     peeked     |   unpeeked       ]
// [      filled      |    unfilled    ]

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers() {
        let mut interner = str::Interner::new();
        let slice = b"  \r\n  xxxxxyyyyyy  asdasd a1337 impl for const";
        let mut reader = PeekReader::with_capacity(4096, &slice[..]);

        let mut lexer = Lexer::new(&mut reader, &mut interner);
        let mut tokens = lexer.into_iter();
        let x = tokens.next();
        let y = tokens.next();
        let z = tokens.next();
        let u = tokens.next();
        let v = tokens.next();
        let w = tokens.next();
        std::assert_matches::assert_matches!(x, Some(Token { kind: TokenKind::Ident(id), .. }) if
            interner.lookup_bytes(id) == b"xxxxxyyyyyy");
        std::assert_matches::assert_matches!(y, Some(Token { kind: TokenKind::Ident(id), ..}) if
            interner.lookup_bytes(id) == b"asdasd");
        std::assert_matches::assert_matches!(z, Some(Token { kind: TokenKind::Ident(id), ..}) if
            interner.lookup_bytes(id) == b"a1337");
        std::assert_matches::assert_matches!(
            u,
            Some(Token {
                kind: TokenKind::Impl,
                ..
            })
        );
        std::assert_matches::assert_matches!(
            v,
            Some(Token {
                kind: TokenKind::For,
                ..
            })
        );
        std::assert_matches::assert_matches!(
            w,
            Some(Token {
                kind: TokenKind::Const,
                ..
            })
        );
    }

    #[test]
    fn prelude() -> std::io::Result<()> {
        let file =
            std::fs::File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/prelude.ge"))?;
        let mut interner = str::Interner::new();
        let mut reader = PeekReader::with_capacity(4096, file);
        let mut lexer = Lexer::new(&mut reader, &mut interner);
        let mut tokens = lexer.into_iter();
        while let Some(token) = tokens.next() {
            println!("{:?}", token);
        }
        Ok(())
    }
}

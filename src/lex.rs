use crate::buffer;
use crate::str;
use std::io::{BufRead, Read};

#[derive(Debug)]
pub enum TokenKind {
    Pound,
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
    OpenBrace,
    CloseBrace,
    Ident(str::Id),
    Literal,
    Lt,
    Gt,
    Eq,
    EqEq,
    Colon,
    Plus,
    Minus,
    Semi,
    Fn,
}

pub struct PeekReader<R: ?Sized> {
    lookahead: usize,
    buf: buffer::Buffer,
    inner: R,
}

impl<R: ?Sized + Read> Read for PeekReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // If we don't have any buffered data and we're doing a massive read
        // (larger than our internal buffer), bypass our internal buffer
        // entirely.
        if self.buf.pos() == self.buf.filled() && buf.len() >= self.buf.capacity() {
            self.buf.discard_buffer();
            return self.inner.read(buf);
        }
        let mut rem = self.fill_buf()?;
        let nread = rem.read(buf)?;
        self.buf.consume(nread);
        Ok(nread)
    }
}

impl<R: ?Sized + Read> BufRead for PeekReader<R> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.buf.fill_buf(&mut self.inner)
    }

    fn consume(&mut self, amt: usize) {
        self.buf.consume(amt);
        self.lookahead = self.lookahead.saturating_sub(amt);
    }
}

impl<R: Read> PeekReader<R>
where
    R: Read,
{
    pub fn with_capacity(capacity: usize, inner: R) -> Self {
        Self {
            inner: inner,
            buf: buffer::Buffer::with_capacity(capacity),
            lookahead: 0usize,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Position {
    pub row: u32,
    pub col: u32,
}

impl<R: Read> PeekAhead for PeekReader<R> {
    type Item = u8;

    fn peek(&self) -> &[Self::Item] {
        &self.buf.buffer()[..self.lookahead()]
    }

    fn lookahead(&self) -> usize {
        self.lookahead
    }

    fn cont(&mut self, amt: usize) -> Result<(), PeekError> {
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

    fn emit(&mut self, kind: TokenKind) -> Option<TokenKind> {
        Some(kind)
    }

    fn identifier(&mut self) -> Option<TokenKind> {
        let id = self
            .inner
            .take_while(|x| is_identifier(x))
            .consume_with(|x| self.interner.intern_bytes(x));
        Some(TokenKind::Ident(id))
    }

    fn literal(&self) -> Option<TokenKind> {
        todo!()
    }

    fn slash(&self) -> Option<TokenKind> {
        todo!()
    }

    fn eqeq(&self) -> Option<TokenKind> {
        todo!()
    }

    fn andand(&self) -> Option<TokenKind> {
        todo!()
    }

    fn oror(&self) -> Option<TokenKind> {
        todo!()
    }
}

fn is_whitespace(byte: &u8) -> bool {
    matches!(*byte, b'\t' | b'\n' | b'\x0C' | b'\r' | b' ')
}

fn is_identifier(byte: &u8) -> bool {
    matches!(*byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9')
}

impl<'a, 'b, R> Iterator for Lexer<'a, 'b, R>
where
    R: Read,
{
    type Item = TokenKind;
    fn next(&mut self) -> Option<Self::Item> {
        let inner = &mut self.inner;
        inner.skip_while(|x| is_whitespace(x));
        match inner.peek_last().ok()? {
            b'/' => self.slash(),
            b'=' => self.eqeq(),
            b'&' => self.andand(),
            b'|' => self.oror(),
            b'-' => self.emit(TokenKind::Minus),
            b':' => self.emit(TokenKind::Colon),
            b';' => self.emit(TokenKind::Semi),
            b'#' => self.emit(TokenKind::Pound),
            b'(' => self.emit(TokenKind::OpenParen),
            b')' => self.emit(TokenKind::CloseParen),
            b'[' => self.emit(TokenKind::OpenBracket),
            b']' => self.emit(TokenKind::CloseBracket),
            b'{' => self.emit(TokenKind::OpenBrace),
            b'}' => self.emit(TokenKind::CloseBrace),
            b'a'..=b'z' | b'A'..=b'Z' => self.identifier(),
            b'0'..=b'9' => self.literal(),
            _ => None,
        }
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

pub trait PeekAhead: Sized + BufRead {
    type Item;

    fn peek(&self) -> &[Self::Item];

    fn lookahead(&self) -> usize;

    fn cont(&mut self, amt: usize) -> Result<(), PeekError>;

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
            self.cont(amt - self.lookahead())?;
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
        T::Item: std::fmt::Debug,
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
    T::Item: std::fmt::Debug,
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
        let slice = b"  \r\n  xxxxxyyyyyy  asdasd a1337";
        let mut reader = PeekReader::with_capacity(4096, &slice[..]);
        let mut lexer = Lexer::new(&mut reader, &mut interner);
        let x = lexer.next();
        let y = lexer.next();
        let z = lexer.next();
        std::assert_matches::assert_matches!(x, Some(TokenKind::Ident(id)) if
            interner.lookup_bytes(id) == b"xxxxxyyyyyy");
        std::assert_matches::assert_matches!(y, Some(TokenKind::Ident(id)) if
            interner.lookup_bytes(id) == b"asdasd");
        std::assert_matches::assert_matches!(z, Some(TokenKind::Ident(id)) if
            interner.lookup_bytes(id) == b"a1337");
    }
}

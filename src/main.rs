use clap::Parser;
use std::fs::File;
use std::io;
use yap::{IntoTokens, Tokens};

#[derive(Parser, Debug)]
#[clap(version)]
struct Args {
    #[clap(short, long, help = "Path to config")]
    path: Option<std::path::PathBuf>,
}
/*
impl Add(x: u64, y: u64) -> u64 {
    __builtin("add", x, y)
}

...connect(add(30))
*/

#[derive(PartialEq, Debug)]
enum Bit {
    Unsigned(u64),
    Signed(u64),
}

#[derive(PartialEq, Debug)]
struct Inputs(std::collections::HashMap<String, Bit>);

#[derive(PartialEq, Debug)]
struct Outputs(std::vec::Vec<Bit>);

#[derive(PartialEq, Debug)]
struct Type(String);

#[derive(PartialEq, Debug)]
enum Ast {
    Module(Type, Inputs, Outputs),
}

#[derive(PartialEq, Debug)]
enum Error {
    InvalidBit,
    InvalidUnsignedWidth,
    InvalidSignedWidth,
    InvalidInputs,
    MissingImpl,
}

fn typ(t: &mut impl Tokens<Item = char>) -> Result<Type, core::convert::Infallible> {
    t.take_while(|t| t.is_alphabetic())
        .parse::<String, String>()
        .map(Type)
}

fn pair(t: &mut impl Tokens<Item = char>) -> Result<(String, Bit), Error> {
    let Ok(s) = t
        .take_while(|c| c.is_alphabetic())
        .parse::<String, String>();
    t.skip_while(|c| c.is_ascii_whitespace());
    t.token(':');
    t.skip_while(|c| c.is_ascii_whitespace());
    let b = bit(&mut *t)?;
    Ok((s, b))
}

fn inputs(t: &mut impl Tokens<Item = char>) -> Result<Inputs, Error> {
    if !t.token('(') {
        return Err(Error::InvalidInputs);
    }
    let inputs: std::collections::HashMap<String, Bit> =
        t.sep_by(|t| pair(t).ok(), |t| field_separator(t)).collect();
    t.token(')');
    Ok(Inputs(inputs))
}

fn parse_digits(t: &mut impl Tokens<Item = char>) -> Result<u64, ()> {
    t.take_while(|c| c.is_digit(10))
        .parse::<u64, String>()
        .map_err(|_| ())
}

fn bit(t: &mut impl Tokens<Item = char>) -> Result<Bit, Error> {
    if t.token('u') {
        return parse_digits(t)
            .map(Bit::Unsigned)
            .map_err(|()| Error::InvalidSignedWidth);
    } else if t.token('s') {
        return parse_digits(t)
            .map(Bit::Signed)
            .map_err(|()| Error::InvalidUnsignedWidth);
    }
    Err(Error::InvalidBit)
}

fn outputs(t: &mut impl Tokens<Item = char>) -> Result<Outputs, Error> {
    let singular = t.token('(');
    let outputs: std::vec::Vec<Bit> = t.sep_by(|t| bit(t).ok(), |t| field_separator(t)).collect();
    if singular {
        t.token(')');
    }
    Ok(Outputs(outputs))
}

fn skip_whitespace(t: &mut impl Tokens<Item = char>) {
    t.skip_while(|c| c.is_ascii_whitespace());
}

fn field_separator(toks: &mut impl Tokens<Item = char>) -> bool {
    toks.surrounded_by(|t| t.token(','), |t| skip_whitespace(t))
}

fn module(t: &mut impl Tokens<Item = char>) -> Result<Ast, Error> {
    if !t.tokens("impl".chars()) {
        return Err(Error::MissingImpl);
    }

    skip_whitespace(&mut *t);
    let Ok(typ) = typ(&mut *t);
    let inputs = inputs(&mut *t)?;
    t.skip_while(|c| c.is_ascii_whitespace());
    t.tokens("->".chars());
    t.skip_while(|c| c.is_ascii_whitespace());
    let outputs = outputs(t)?;
    Ok(Ast::Module(typ, inputs, outputs))
}

fn compile<R>(mut r: R) -> io::Result<()>
where
    R: io::BufRead,
{
    let mut buf = Vec::<u8>::new();
    r.read_to_end(&mut buf)?;
    let buf = unsafe { String::from_utf8_unchecked(buf) };

    let mut tokens = buf.as_str().into_tokens();
    let module = module(&mut tokens).map_err(|s| io::Error::other(format!("{:?}", s)))?;
    println!("{:?}", module);
    println!("{}", tokens.remaining());

    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    if let Some(filename) = args.path {
        let file = File::open(filename)?;
        compile(io::BufReader::new(file))?;
    } else {
        compile(io::stdin().lock())?;
    }
    Ok(())
}

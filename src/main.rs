use clap::Parser;
use melior::{
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        operation::{OperationBuilder, OperationPrintingFlags},
        *,
    },
    Context,
};
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

impl Type {
    fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(PartialEq, Debug)]
enum Expression {
    Builtin(String, std::vec::Vec<String>),
}

#[derive(PartialEq, Debug)]
struct Body(std::vec::Vec<Expression>);

#[derive(PartialEq, Debug)]
enum Ast {
    Module(Type, Inputs, Outputs, Body),
}

#[derive(PartialEq, Debug)]
enum Error {
    InvalidBit,
    InvalidUnsignedWidth,
    InvalidSignedWidth,
    InvalidInputs,
}

fn alphastr(t: &mut impl Tokens<Item = char>) -> String {
    let Ok(str) = t
        .take_while(|t| t.is_alphabetic())
        .parse::<String, String>();
    str
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
    let inputs: std::collections::HashMap<String, Bit> = t
        .sep_by(|t| pair(t).ok(), |t| field_separator(',', t))
        .collect();
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
    let outputs: std::vec::Vec<Bit> = t
        .sep_by(|t| bit(t).ok(), |t| field_separator(',', t))
        .collect();
    if singular {
        t.token(')');
    }
    Ok(Outputs(outputs))
}

fn expr_builtin(t: &mut impl Tokens<Item = char>) -> Option<Result<Expression, Error>> {
    if !t.tokens("__builtin".chars()) {
        return None;
    }
    Some(builtin(t))
}

fn builtin(t: &mut impl Tokens<Item = char>) -> Result<Expression, Error> {
    t.token('(');
    t.token('"');
    let Ok(str) = t.take_while(|&c| c != '"').parse::<String, String>();
    t.token('"');
    field_separator(',', t);
    let args: std::vec::Vec<String> = t
        .sep_by(|t| Some(alphastr(t)), |t| field_separator(',', t))
        .collect();

    t.token(')');
    Ok(Expression::Builtin(str, args))
}

fn body(t: &mut impl Tokens<Item = char>) -> Result<Body, Error> {
    t.sep_by(|t| expr_builtin(t), |t| field_separator(';', t))
        .collect::<Result<std::vec::Vec<Expression>, Error>>()
        .map(Body)
}

fn skip_whitespace(t: &mut impl Tokens<Item = char>) {
    t.skip_while(|c| c.is_ascii_whitespace());
}

fn field_separator(separator: char, toks: &mut impl Tokens<Item = char>) -> bool {
    toks.surrounded_by(|t| t.token(separator), |t| skip_whitespace(t))
}

fn impl_module(t: &mut impl Tokens<Item = char>) -> Option<Result<Ast, Error>> {
    if !t.tokens("impl".chars()) {
        return None;
    }
    Some(module(t))
}

fn module(t: &mut impl Tokens<Item = char>) -> Result<Ast, Error> {
    skip_whitespace(t);
    let typ = Type(alphastr(t));
    let inputs = inputs(t)?;
    t.skip_while(|c| c.is_ascii_whitespace());
    t.tokens("->".chars());
    t.skip_while(|c| c.is_ascii_whitespace());
    let outputs = outputs(t)?;
    t.skip_while(|c| c.is_ascii_whitespace());
    t.token('{');
    t.skip_while(|c| c.is_ascii_whitespace());
    let body = body(t)?;
    t.skip_while(|c| c.is_ascii_whitespace());
    t.token('}');
    Ok(Ast::Module(typ, inputs, outputs, body))
}

fn parse(str: &str) -> Result<std::vec::Vec<Ast>, Error> {
    str.into_tokens()
        .sep_by(
            |t| impl_module(t),
            |t| t.skip_while(|c| c.is_ascii_whitespace()) != 0usize,
        )
        .collect::<Result<std::vec::Vec<Ast>, Error>>()
}

fn emit<'c>(context: &'c Context, node: &Ast) -> Operation<'c> {
    let Ast::Module(typ, inputs, outputs, body) = node;

    let arguments = Vec::<(Type, Location)>::with_capacity(inputs.0.len() + outputs.0.len());
    for input in &inputs.0 {}
    for output in &outputs.0 {}

    OperationBuilder::new("hw.module", Location::unknown(&context))
        .add_attributes(&[(
            Identifier::new(context, "sym_name"),
            StringAttribute::new(&context, typ.as_str()).into(),
        )])
        .build()
        .expect("valid hw.module")
}

fn compile<R>(mut r: R) -> io::Result<()>
where
    R: io::BufRead,
{
    let mut buf = Vec::<u8>::new();
    r.read_to_end(&mut buf)?;
    let buf = unsafe { String::from_utf8_unchecked(buf) };

    let ast = parse(buf.as_str()).map_err(|e| io::Error::other(format!("{:?}", e)))?;

    let context = Context::new();
    context.set_allow_unregistered_dialects(true);

    let module = Module::new(Location::unknown(&context));
    for node in ast {
        module.body().append_operation(emit(&context, &node));
    }

    let flags = OperationPrintingFlags::default();
    let text = module
        .as_operation()
        .to_string_with_flags(flags)
        .map_err(|e| io::Error::other(format!("{}", e)))?;
    println!("{}", text);

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

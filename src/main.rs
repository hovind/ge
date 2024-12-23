use clap::Parser;
use melior::{
    ir::{
        attribute::{ArrayAttribute, StringAttribute, TypeAttribute},
        operation::{OperationBuilder, OperationPrintingFlags},
        r#type::IntegerType,
        Type, *,
    },
    Context,
};
use std::collections::HashMap;
use std::convert::Into;
use std::fs::File;
use std::io;
use std::vec::Vec;
use yap::{IntoTokens, Tokens};

pub mod scope;
pub mod str;

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

#[derive(Clone, Copy, Debug, PartialEq)]
enum Bit {
    Unsigned(u32),
    Signed(u32),
}

impl Bit {
    fn width(self) -> u32 {
        match self {
            Bit::Unsigned(w) => w,
            Bit::Signed(w) => w,
        }
    }
}

#[derive(Debug, PartialEq)]
struct Inputs(HashMap<str::Id, Bit>);

#[derive(Debug, PartialEq)]
struct Outputs(std::vec::Vec<Bit>);

#[derive(Debug, PartialEq)]
struct Name(str::Id);

#[derive(Debug, PartialEq)]
enum Expression {
    Builtin(str::Id, std::vec::Vec<str::Id>),
}

#[derive(Debug, PartialEq)]
struct Body(std::vec::Vec<Expression>);

#[derive(Debug, PartialEq)]
enum Ast {
    Module(Name, Inputs, Outputs, Body),
}

#[derive(Debug, PartialEq)]
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

fn pair(i: &mut str::Interner, t: &mut impl Tokens<Item = char>) -> Result<(str::Id, Bit), Error> {
    let Ok(s) = t
        .take_while(|c| c.is_alphabetic())
        .parse::<String, String>();
    t.skip_while(|c| c.is_ascii_whitespace());
    t.token(':');
    t.skip_while(|c| c.is_ascii_whitespace());
    let b = bit(&mut *t)?;
    Ok((i.intern(&s), b))
}

fn inputs(i: &mut str::Interner, t: &mut impl Tokens<Item = char>) -> Result<Inputs, Error> {
    if !t.token('(') {
        return Err(Error::InvalidInputs);
    }
    let inputs: HashMap<str::Id, Bit> = t
        .sep_by(|t| pair(i, t).ok(), |t| field_separator(',', t))
        .collect();
    t.token(')');
    Ok(Inputs(inputs))
}

fn parse_digits(t: &mut impl Tokens<Item = char>) -> Result<u32, ()> {
    t.take_while(|c| c.is_digit(10))
        .parse::<_, String>()
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

fn expr_builtin(
    i: &mut str::Interner,
    t: &mut impl Tokens<Item = char>,
) -> Option<Result<Expression, Error>> {
    if !t.tokens("__builtin".chars()) {
        return None;
    }
    Some(builtin(i, t))
}

fn builtin(i: &mut str::Interner, t: &mut impl Tokens<Item = char>) -> Result<Expression, Error> {
    t.token('(');
    t.token('"');
    let Ok(str) = t.take_while(|&c| c != '"').parse::<String, String>();
    let str = i.intern(&str);
    t.token('"');
    field_separator(',', t);
    let args: std::vec::Vec<str::Id> = t
        .sep_by(
            |t| Some(i.intern(&alphastr(t))),
            |t| field_separator(',', t),
        )
        .collect();

    t.token(')');
    Ok(Expression::Builtin(str, args))
}

fn body(i: &mut str::Interner, t: &mut impl Tokens<Item = char>) -> Result<Body, Error> {
    t.sep_by(|t| expr_builtin(i, t), |t| field_separator(';', t))
        .collect::<Result<std::vec::Vec<Expression>, Error>>()
        .map(Body)
}

fn skip_whitespace(t: &mut impl Tokens<Item = char>) {
    t.skip_while(|c| c.is_ascii_whitespace());
}

fn field_separator(separator: char, toks: &mut impl Tokens<Item = char>) -> bool {
    toks.surrounded_by(|t| t.token(separator), |t| skip_whitespace(t))
}

fn impl_module(
    i: &mut str::Interner,
    t: &mut impl Tokens<Item = char>,
) -> Option<Result<Ast, Error>> {
    if !t.tokens("impl".chars()) {
        return None;
    }
    Some(module(i, t))
}

fn module(i: &mut str::Interner, t: &mut impl Tokens<Item = char>) -> Result<Ast, Error> {
    skip_whitespace(t);
    let typ = Name(i.intern(&alphastr(t)));
    let inputs = inputs(i, t)?;
    t.skip_while(|c| c.is_ascii_whitespace());
    t.tokens("->".chars());
    t.skip_while(|c| c.is_ascii_whitespace());
    let outputs = outputs(t)?;
    t.skip_while(|c| c.is_ascii_whitespace());
    t.token('{');
    t.skip_while(|c| c.is_ascii_whitespace());
    let body = body(i, t)?;
    t.skip_while(|c| c.is_ascii_whitespace());
    t.token('}');
    Ok(Ast::Module(typ, inputs, outputs, body))
}

fn parse(i: &mut str::Interner, str: &str) -> Result<std::vec::Vec<Ast>, Error> {
    str.into_tokens()
        .sep_by(
            |t| impl_module(i, t),
            |t| t.skip_while(|c| c.is_ascii_whitespace()) != 0usize,
        )
        .collect::<Result<std::vec::Vec<Ast>, Error>>()
}

fn emit<'c>(context: &'c Context, i: &str::Interner, node: &Ast) -> Operation<'c> {
    let Ast::Module(typ, inputs, _outputs, body) = node;
    let mut scope = scope::Ctx::<'c, '_, str::Id>::new();

    let mut arguments = Vec::<(Type, Location)>::with_capacity(inputs.0.len());
    for (_, bit) in &inputs.0 {
        arguments.push((
            Type::from(IntegerType::new(&context, bit.width())),
            Location::unknown(&context),
        ));
    }

    let block = Block::new(&arguments);
    for (i, id) in inputs.0.keys().enumerate() {
        let arg = block.argument(i).expect("valid block argument");
        scope.insert(*id, arg.into());
    }

    for Expression::Builtin(name, args) in &body.0 {
        let mut operands = Vec::<Value<'c, '_>>::new();
        for id in args.iter() {
            let arg = scope.get(*id).expect("failed to resolve identifier");
            operands.push(arg);
        }
        let op = OperationBuilder::new(i.lookup(*name), Location::unknown(&context))
            .add_operands(&operands)
            .build()
            .unwrap();
        block.append_operation(op);
    }
    let region = Region::new();
    region.append_block(block);

    let module_type = Type::parse(&context, "!hw.modty<>").unwrap();
    OperationBuilder::new("hw.module", Location::unknown(&context))
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(&context, i.lookup(typ.0)).into(),
            ),
            (
                Identifier::new(context, "module_type"),
                TypeAttribute::new(module_type).into(),
            ),
            (
                Identifier::new(context, "parameters"),
                ArrayAttribute::new(&context, &[]).into(),
            ),
        ])
        .add_regions([region])
        .build()
        .unwrap()
}

fn compile<R>(mut r: R) -> io::Result<()>
where
    R: io::BufRead,
{
    let mut buf = Vec::<u8>::new();
    r.read_to_end(&mut buf)?;
    let buf = unsafe { String::from_utf8_unchecked(buf) };

    let mut interner = str::Interner::new();
    let ast =
        parse(&mut interner, buf.as_str()).map_err(|e| io::Error::other(format!("{:?}", e)))?;

    let context = Context::new();
    context.set_allow_unregistered_dialects(true);

    let module = Module::new(Location::unknown(&context));
    for node in ast {
        module
            .body()
            .append_operation(emit(&context, &interner, &node));
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

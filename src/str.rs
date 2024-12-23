use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Id(u32);

#[derive(Default)]
pub struct Interner {
    map: HashMap<String, Id>,
    vec: Vec<String>,
}

impl Interner {
    pub fn new() -> Interner {
        Interner {
            map: HashMap::default(),
            vec: Vec::new(),
        }
    }

    pub fn intern(&mut self, name: &str) -> Id {
        if let Some(&idx) = self.map.get(name) {
            return idx;
        }
        let idx = Id(self.map.len() as u32);
        self.map.insert(name.to_owned(), idx);
        self.vec.push(name.to_owned());

        debug_assert!(self.lookup(idx) == name);
        debug_assert!(self.intern(name) == idx);

        idx
    }

    pub fn lookup(&self, idx: Id) -> &str {
        self.vec[idx.0 as usize].as_str()
    }
}

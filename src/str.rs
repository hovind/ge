use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Id(u32);

#[derive(Default)]
pub struct Interner {
    map: HashMap<Vec<u8>, Id>,
    vec: Vec<Vec<u8>>,
}

impl Interner {
    pub fn new() -> Interner {
        Interner {
            map: HashMap::default(),
            vec: Vec::new(),
        }
    }

    pub fn intern_bytes(&mut self, name: &[u8]) -> Id {
        if let Some(&idx) = self.map.get(name) {
            return idx;
        }
        let idx = Id(self.map.len() as u32);
        self.map.insert(name.to_owned(), idx);
        self.vec.push(name.to_owned());

        debug_assert!(self.lookup_bytes(idx) == name);
        debug_assert!(self.intern_bytes(name) == idx);

        idx
    }

    pub fn lookup_bytes(&self, idx: Id) -> &[u8] {
        &self.vec[idx.0 as usize]
    }

    pub fn intern(&mut self, name: &str) -> Id {
        self.intern_bytes(name.as_bytes())
    }

    pub fn lookup(&self, idx: Id) -> &str {
        unsafe { std::str::from_utf8_unchecked(self.lookup_bytes(idx)) }
    }
}

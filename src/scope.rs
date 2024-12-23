use melior::ir::Value;
use std::collections::BTreeMap;

pub struct Ctx<'c, 'v, K> {
    vals: BTreeMap<(u32, K), Value<'c, 'v>>,
    depth: u32,
}

impl<'c, 'v, K> Ctx<'c, 'v, K>
where
    K: Ord,
{
    pub fn new() -> Ctx<'c, 'v, K> {
        Ctx {
            vals: BTreeMap::new(),
            depth: 0,
        }
    }

    pub fn push(&mut self) {
        self.depth += 1;
    }

    pub fn pop(&mut self) {
        self.vals.retain(|&(d, _), _| d < self.depth);
        self.depth -= 1;
    }

    pub fn insert(&mut self, key: K, value: Value<'c, 'v>) -> Option<Value<'c, 'v>> {
        self.vals.insert((self.depth, key), value)
    }

    pub fn get(&self, key: K) -> Option<Value<'c, 'v>> {
        self.vals.get(&(self.depth, key)).copied()
    }
}

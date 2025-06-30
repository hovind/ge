mod arc {
    mod sim {
        struct Instance<T>(std::marker::PhantomData<T>);

        fn instantiate<F, Mod>(module: F)
        where
            F: FnOnce(Instance<Mod>) -> (),
            Mod: Module,
        {
            todo!()
        }

        trait Module {
            type Args;
            type Output;

            fn new() -> Self;
            fn eval(&mut self, args: Self::Args) -> Self::Output;
        }

        #[test]
        fn main() {
            arc::sim::instantiate::<Counter>(|x| {
                x.clk.poke(0);
                x.step();
            })
        }

        struct Counter<const MAX: usize> {
            count: u32,
        }

        impl<const MAX: usize> Module for Counter<MAX> {
            type Args = ();
            type Output = u32;

            fn new() -> Self {
                Counter { count: 0 }
            }

            fn eval(&mut self, args: Self::Args) -> Self::Output {
                self.count = if self.count as usize == MAX {
                    self.count + 1
                } else {
                    0
                };
                self.count
            }
        }
    }
}

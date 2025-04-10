pub trait Policy<S, A> {
    // 根据状态选择动作
    fn select_action(&self, state: &S) -> A;

    // // 探索模式下选择动作(可选)
    // fn explore(&self, state: &S) -> A {
    //     self.select_action(state)
    // }
}

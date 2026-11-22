pub trait Policy<S, A> {
    /// 根据状态选择动作
    fn select_action(&self, state: &S) -> A;
}

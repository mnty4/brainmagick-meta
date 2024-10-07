class MetaLearner(nn.Module):
    def __init__(self, net_class, net_class_args, n_way, k_shot, meta_batchesz, beta, num_updates, num_updates_test) -> None:
        super(MetaLearner, self).__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.meta_batchesz = meta_batchesz
        self.beta = beta
        self.num_updates = num_updates
        self.num_updates_test = num_updates_test

        self.learner = Learner(net_class, *net_class_args)
        self.optimizer = optim.Adam(self.learner.parameters(), lr=beta)

    def write_grads(self, dummy_loss, sum_grads_pi):
        hooks = []
        for i, v in enumerate(self.learner.parameters()):
            def closure():
                ii = i
                return lambda grad : sum_grads_pi[ii]
            # h = v.register_hook(closure())
            hooks.append(v.register_hook(closure()))

        self.optimizer.zero_grad()
        dummy_loss.backward()
        self.optimizer.step()

        for h in hooks:
            h.remove()

    def forward(self, support_x, support_y, query_x, query_y):
        sum_grads_pi = None
        meta_batchesz = support_y.size(0)

        accs = []
        for i in range(meta_batchesz):
            _, grad_pi, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            accs.append(episode_acc)
            if sum_grads_pi is None:
                sum_grads_pi = grad_pi
            else:
                sum_grads_pi = [torch.add(p, q) for p, q in zip(sum_grads_pi, grad_pi)]
        dummy_loss, _ = self.learner.net_forward(support_x[0], support_y[0])
        self.write_grads(dummy_loss, sum_grads_pi)

        return accs

    def pred(self, support_x, support_y, query_x, query_y):
        meta_batchesz = support_y.size(0)
        accs = []
        for i in range(meta_batchesz):
            _, _, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates_test, testing=True)
            accs.append(episode_acc)
        return np.array(accs).mean()
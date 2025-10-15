from src.solver import build_solver_from_sources
model, artifacts, _, _ = build_solver_from_sources('config.yaml', 'data_fake_month')
proto = model.Proto()
count = 0
for i, ct in enumerate(proto.constraints):
    if ct.HasField('linear'):
        domain = list(ct.linear.domain)
        if domain and domain[-1] == -1:
            print('idx', i, 'vars', list(ct.linear.vars), 'coeffs', list(ct.linear.coeffs), 'domain', domain)
            count += 1
            if count >= 5:
                break

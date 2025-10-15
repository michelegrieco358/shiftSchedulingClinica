from src.solver import build_solver_from_sources
model, _, _, _ = build_solver_from_sources('config.yaml', 'data_fake_month')
model.ExportToFile('model_fake.pbtxt')

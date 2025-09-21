import h5py
p = r'c:\Users\sarus\Downloads\ASL Translator\keras_model.h5'
print('Inspecting', p)
try:
    with h5py.File(p, 'r') as f:
        print('Top keys:', list(f.keys()))
        for k,v in f.attrs.items():
            print('ATTR:', k, type(v), v)
        # common attrs
        kers = f.attrs.get('keras_version') or f.attrs.get('keras_version'.encode())
        backend = f.attrs.get('backend') or f.attrs.get('backend'.encode())
        if kers:
            print('keras_version attr:', kers)
        if backend:
            print('backend attr:', backend)
        if 'model_config' in f.attrs:
            cfg = f.attrs['model_config']
            if isinstance(cfg, bytes):
                cfg = cfg.decode('utf-8')
            print('\nmodel_config contains groups?','"groups"' in cfg)
except Exception as e:
    print('ERR', e)

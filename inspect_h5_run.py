import h5py, json, traceback
p = r'c:\Users\sarus\Downloads\ASL Translator\keras_model.h5'
print('MODEL_PATH=', p)
try:
    with h5py.File(p, 'r') as f:
        print('Top keys:', list(f.keys()))
        for k, v in f.attrs.items():
            print('ATTR:', k, type(v), repr(v) if isinstance(v, (str, bytes)) else v)
        if 'model_config' in f.attrs:
            cfg = f.attrs['model_config']
            if isinstance(cfg, bytes):
                cfg = cfg.decode('utf-8')
            print('\nmodel_config length:', len(cfg))
            print('\nContains "groups"?', '"groups"' in cfg)
            print('\nPreview first 1000 chars:\n')
            print(cfg[:1000])
except Exception as e:
    print('ERR', e)
    traceback.print_exc()

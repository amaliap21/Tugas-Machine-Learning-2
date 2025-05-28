from layers.layer import Layer
import numpy as np
import h5py

class Sequential:
    def __init__(self, layers=None):
        self.layers: list[Layer] = []

        if layers:
            key_count = {}
            for layer in layers:
                key = getattr(layer, "key", None)
                if key:
                    key_count[key] = key_count.get(key, 0) + 1
            duplicates_exist = any(count > 1 for count in key_count.values())
            if duplicates_exist:
                rename_counter = {}
                for layer in layers:
                    key = getattr(layer, "key", None)
                    if key:
                        rename_counter[key] = rename_counter.get(key, 0) + 1
                        new_key = f"{key}_{rename_counter[key]}"
                        layer.key = new_key
            self.layers.extend(layers)

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("Layer must be an instance of Layer class.")
        self.layers.append(layer)

    def predict(self, initial_x):
        next_x = initial_x
        for layer in self.layers:
            next_x = layer.forward(next_x)
        return next_x

    def load_weights(self, path_to_h5_file: str):
        with h5py.File(path_to_h5_file, 'r') as f:
            if 'model_weights' not in f:
                print(f"Error: 'model_weights' group not found in HDF5 file: {path_to_h5_file}")
                return
            keras_weights_group = f['model_weights']
            keras_layer_names_in_file = list(keras_weights_group.keys())
            for i, custom_layer in enumerate(self.layers):
                matched = False

                if not hasattr(custom_layer, "key"):
                    print(f"Custom layer {i} (type: {type(custom_layer).__name__}) has no 'key' attribute, skipping.")
                    continue
                
                print(keras_layer_names_in_file)
                if custom_layer.key in keras_layer_names_in_file:
                    try:
                        keras_layer_h5_group = keras_weights_group[custom_layer.key]
                        weights_for_this_layer = []
                        for member_name in keras_layer_h5_group.keys():
                            h5_item = keras_layer_h5_group.get(member_name)
                            if h5_item is not None:
                                if isinstance(h5_item, h5py.Dataset):
                                    weights_for_this_layer.append(h5_item[()])
                        if not weights_for_this_layer and hasattr(keras_layer_h5_group, 'attrs') and 'weight_names' in keras_layer_h5_group.attrs:
                            print(f"INFO: Layer '{custom_layer.key}' has 'weight_names' attribute. Attempting to load based on 'weight_names'.")
                            raw_weight_names_from_attr = keras_layer_h5_group.attrs['weight_names']
                            processed_weight_names = []
                            for name_in_attr in raw_weight_names_from_attr:
                                if isinstance(name_in_attr, bytes):
                                    processed_weight_names.append(name_in_attr.decode('utf-8'))
                                elif isinstance(name_in_attr, str):
                                    processed_weight_names.append(name_in_attr)
                                else:
                                    print(f"WARNING: Item '{name_in_attr}' in 'weight_names' for layer '{custom_layer.key}' "
                                          f"has unexpected type {type(name_in_attr)}. Skipping.")
                                    continue
                            for name_to_load in processed_weight_names:
                                dataset_to_load = keras_layer_h5_group.get(name_to_load)
                                if isinstance(dataset_to_load, h5py.Dataset):
                                    weights_for_this_layer.append(dataset_to_load[()])
                                else:
                                    print(f"WARNING: Weight '{name_to_load}' listed in 'weight_names' for layer "
                                          f"'{custom_layer.key}' was not found or is not a Dataset directly within its group "
                                          f"(found type: {type(dataset_to_load)}).")
                        if not weights_for_this_layer and custom_layer.key in ["embedding", "dense", "simple_rnn"]:
                             print(f"WARNING: No weights loaded for layer '{custom_layer.key}'. There may be no loadable datasets in the HDF5 group or the structure may not be what you expect.")
                        print(f"Loading {len(weights_for_this_layer)} weight array(s) from HDF5 group '{custom_layer.key}' into custom layer {i} ({type(custom_layer).__name__})")
                        custom_layer.load_keras_weights(weights_for_this_layer)
                        matched = True
                    except Exception as e:
                        print(f"Error loading weights for custom layer {i} (key='{custom_layer.key}', type={type(custom_layer).__name__}): {e}")
                else:
                    print(f"No matching Keras weights found in HDF5 file for custom layer {i} (key='{custom_layer.key}', type={type(custom_layer).__name__})")

                if not matched:
                    print(f"Skipped loading weights for custom layer {i} (key='{custom_layer.key}', type={type(custom_layer).__name__})")
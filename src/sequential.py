from layers.layer import Layer
import numpy as np
import h5py

class Sequential:
    def __init__(self, layers=None):
        self.layers: list[Layer] = []
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("Layer must be an instance of Layer class.")
        self.layers.append(layer)

    def predict(self, initial_x):
        if hasattr(initial_x, 'numpy'):
            next_x = initial_x.numpy()
        else:
            next_x = np.array(initial_x)
        mask = None
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'mask_zero') and layer.mask_zero:
                result = layer.forward(next_x)
                if isinstance(result, tuple):
                    next_x, mask = result
                else:
                    next_x = result
            elif hasattr(layer, 'forward') and 'mask' in layer.forward.__code__.co_varnames:
                next_x = layer.forward(next_x, mask=mask)
            else:
                next_x = layer.forward(next_x)
                
        return next_x

    def load_weights(self, path_to_h5_file: str):
        import re
        from collections import defaultdict

        with h5py.File(path_to_h5_file, 'r') as f:
            if 'model_weights' not in f:
                print(f"Error: 'model_weights' group not found in HDF5 file: {path_to_h5_file}")
                return
            
            keras_weights_group = f['model_weights']
            keras_layer_names_in_file = list(keras_weights_group.keys())
            print("HDF5 layer keys:", keras_layer_names_in_file)

            # Build a mapping from base_key → list of available indexed variants
            available_layers = defaultdict(list)
            pattern = re.compile(r"^(.*?)(?:_(\d+))?$")
            for name in keras_layer_names_in_file:
                match = pattern.match(name)
                if match:
                    base_key = match.group(1)
                    available_layers[base_key].append(name)

            # Track which index to use for each base_key
            usage_counter = defaultdict(int)

            for i, custom_layer in enumerate(self.layers):
                matched = False

                if not hasattr(custom_layer, "key"):
                    print(f"Custom layer {i} (type: {type(custom_layer).__name__}) has no 'key' attribute, skipping.")
                    continue

                base_key = custom_layer.key
                usage_counter[base_key] += 1
                current_index = usage_counter[base_key] - 1  # 0-based index

                # Get the correct key variant to match
                candidate_names = sorted(available_layers.get(base_key, []))
                if not candidate_names:
                    print(f"No matching group in file for layer key base '{base_key}'")
                    continue

                # Choose either the base name or indexed variant
                if current_index < len(candidate_names):
                    keras_key = candidate_names[current_index]
                else:
                    print(f"Index {current_index} out of range for layer key '{base_key}'")
                    continue

                try:
                    keras_layer_h5_group = keras_weights_group[keras_key]
                    weights_for_this_layer = []

                    for member_name in keras_layer_h5_group.keys():
                        h5_item = keras_layer_h5_group.get(member_name)
                        if isinstance(h5_item, h5py.Dataset):
                            weights_for_this_layer.append(h5_item[()])

                    # Fallback if dataset is under weight_names attribute
                    if not weights_for_this_layer and 'weight_names' in keras_layer_h5_group.attrs:
                        raw_names = keras_layer_h5_group.attrs['weight_names']
                        for raw in raw_names:
                            name = raw.decode('utf-8') if isinstance(raw, bytes) else raw
                            data = keras_layer_h5_group.get(name)
                            if isinstance(data, h5py.Dataset):
                                weights_for_this_layer.append(data[()])

                    if not weights_for_this_layer:
                        print(f"WARNING: No weights found for layer '{keras_key}'.")

                    print(f"Loading {len(weights_for_this_layer)} weight array(s) into layer {i} (type: {type(custom_layer).__name__}, key: {keras_key})")
                    custom_layer.load_keras_weights(weights_for_this_layer)
                    matched = True
                except Exception as e:
                    print(f"Error loading weights for custom layer {i} (key='{keras_key}'): {e}")

                if not matched:
                    print(f"Skipped loading weights for custom layer {i} (original key='{custom_layer.key}', resolved='{keras_key}')")
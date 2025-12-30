import numpy as np

def analyze_data(file_path):
    try:
        # Read headers and data separately
        headers = np.genfromtxt(file_path, delimiter=',', dtype=str, encoding='utf-8', max_rows=1, filling_values='')
        data = np.genfromtxt(file_path, delimiter=',', dtype=str, encoding='utf-8', skip_header=1, filling_values='')

        # Transpose data for column-wise access
        if data.ndim == 1:
            data = data.reshape(1, -1)
        columns_data = data.T

        dtype_list = []
        missing_info = []
        outliers_anomalies_info = []
        total_rows = len(data)

        for i, col_name in enumerate(headers):
            col_values = columns_data[i]

            is_integer = True
            is_float = True
            missing_count = 0
            max_len = 0

            for val in col_values:
                val = val.strip()
                current_len = len(val)

                if current_len > max_len:
                    max_len = current_len

                # Check for missing values
                if val == '' or val == 'nan':
                    missing_count += 1
                    continue

                # Check for Integer
                if is_integer:
                    try:
                        int(val)
                    except ValueError:
                        is_integer = False

                # Check for Float
                if is_float:
                    try:
                        float(val)
                    except ValueError:
                        is_float = False

            # Determine dtype
            if is_integer and missing_count == 0:
                dtype = 'i8'
            elif is_float or (is_integer and missing_count > 0):
                dtype = 'f8'
            else:
                dtype = f'U{max_len}'
            dtype_list.append((col_name, dtype))

            # Missing values analysis
            missing_rate = (missing_count / total_rows) * 100
            if missing_count > 0:
                missing_info.append({
                    'column': col_name,
                    'missing_count': missing_count,
                    'missing_rate': f'{missing_rate:.2f}%'
                })

            # Outliers & Anomalies Analysis
            non_empty = col_values[(col_values != '') & (col_values != 'nan')]
            if len(non_empty) == 0:
                continue

            unique_values = np.unique(non_empty)
            try:
                numeric_values = non_empty.astype(float)

                # Calculate Q1, Q3, IQR outliers
                q1 = np.percentile(numeric_values, 25)
                q3 = np.percentile(numeric_values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = numeric_values[(numeric_values < lower_bound) | (numeric_values > upper_bound)]
                if len(outliers) > 0:
                    outliers_anomalies_info.append({
                        'col_name': col_name,
                        'dtype': 'Numerical',
                        'outliers_count': len(outliers),
                        'outliers_range': (lower_bound, upper_bound),
                        'outliers': outliers
                    })


            except ValueError:
                value_counts = {}
                for val in non_empty:
                    value_counts[val] = value_counts.get(val, 0) + 1

                # Find rare values (less than 1% of total non-missing entries)
                rare_threshold = len(non_empty) * 0.01
                rare_values = {k: v for k, v in value_counts.items() if v < rare_threshold}

                if rare_values:
                    outliers_anomalies_info.append({
                        'col_name': col_name,
                        'dtype': 'Categorical',
                        'unique_values_count': len(unique_values),
                        'rare_values_count': len(rare_values),
                        'rare_values': rare_values,
                    })

        # Dulicate rows analysis
        unique_rows = set([tuple(row) for row in data])
        unique_count = len(unique_rows)

        duplicate_count = total_rows - unique_count
        duplicate_rate = (duplicate_count / total_rows) * 100

        # Informative prints
        print(f'====== Intitalizing file: {file_path} ======')
        print(f'  - Headers: {headers}\n')
        print(f'  - Shape: {data.shape}\n')
        print(f'  - Data types: {dtype_list}\n')
        print(f'  - First 5 rows of data:\n {data[:5]}\n')
        print(f"  - Duplicate rows: {duplicate_count} ({duplicate_rate:.2f}%)\n")

        print(f'  - Missing Values Analysis:')
        if missing_info:
            for info in missing_info:
                print(f"    + {info['column']}: {info['missing_count']} ({info['missing_rate']})")
        else:
            print("    + No missing values found!")
        print('')

        print(f'  - Outliers & Anomalies Analysis:')
        if outliers_anomalies_info:
            for info in outliers_anomalies_info:
                if info['dtype'] == 'Numerical':
                    print(f"  + {info['col_name']} (Numerical):")
                    print(f"    * Outliers count: {info['outliers_count']}")
                    print(f"    * Range: [{info['outliers_range'][0]:.2f}, {info['outliers_range'][1]:.2f}]")
                    print(f"    * Min outlier: {info['outliers'].min():.2f}, Max outlier: {info['outliers'].max():.2f}")
                else:
                    print(f"  + {info['col_name']} (Categorical):")
                    print(f"    * Unique values: {info['unique_values_count']}")
                    print(f"    * Rare values (<1%): {info['rare_values_count']}")
                    for val, count in list(info['rare_values'].items())[:5]:
                        print(f"      * '{val}': {count} times")

        print('=============================================\n')
        return data, headers

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
def load_data(train_path, test_path):
    """Loads the training and test data from CSV files."""
    train_data = np.genfromtxt(train_path, delimiter=',', skip_header=1,
                               dtype=[('enrollee_id', 'i8'), ('city', 'U8'),
                                      ('city_development_index', 'f8'), ('gender', 'U6'),
                                      ('relevent_experience', 'U23'), ('enrolled_university', 'U16'),
                                      ('education_level', 'U14'), ('major_discipline', 'U15'),
                                      ('experience', 'U3'), ('company_size', 'U9'),
                                      ('company_type', 'U19'), ('last_new_job', 'U5'),
                                      ('training_hours', 'i8'), ('target', 'f8')])
    test_data = np.genfromtxt(test_path, delimiter=',', skip_header=1,
                              dtype=[('enrollee_id', 'i8'), ('city', 'U8'),
                                     ('city_development_index', 'f8'), ('gender', 'U6'),
                                     ('relevent_experience', 'U23'), ('enrolled_university', 'U16'),
                                     ('education_level', 'U14'), ('major_discipline', 'U15'),
                                     ('experience', 'U3'), ('company_size', 'U9'),
                                     ('company_type', 'U19'), ('last_new_job', 'U5'),
                                     ('training_hours', 'i8')])
    return train_data, test_data

def downsample(data, target_column='target'):
    """Downsamples the majority class to balance the dataset."""
    # Separate the majority and minority classes
    minority_class = data[data[target_column] == 1]
    majority_class = data[data[target_column] == 0]

    # Get the number of samples in the minority class
    minority_size = len(minority_class)

    # Randomly select a subset of the majority class
    majority_downsampled_indices = np.random.choice(len(majority_class), size=minority_size, replace=False)
    majority_downsampled = majority_class[majority_downsampled_indices]

    # Concatenate the downsampled majority class with the minority class
    downsampled_data = np.concatenate([majority_downsampled, minority_class])

    # Shuffle the data
    np.random.shuffle(downsampled_data)

    return downsampled_data

def get_mode(array):
    """Calculates the mode of a NumPy array, ignoring empty strings."""
    unique_elements, counts = np.unique(array[array != ''], return_counts=True)
    return unique_elements[np.argmax(counts)]

def fill_missing_values(data, is_train=True, mode_values=None):
    """Fills missing values (empty strings) in the dataset using the mode."""
    data_filled = np.copy(data)

    if is_train:
        mode_values = {}
    for name in ['enrolled_university', 'education_level', 'last_new_job', 'experience']:
        if is_train:
            unique_values, counts = np.unique(data[name][data[name] != ''], return_counts=True)
            mode = unique_values[np.argmax(counts)]
            mode_values[name] = mode
        else:
            mode = mode_values[name]
        data_filled[name][data_filled[name] == ''] = mode

    for name in ['company_size', 'company_type', 'major_discipline', 'gender']:
        data_filled[name][data_filled[name] == ''] = 'Unknown'

    return data_filled, mode_values

def encode_and_scale(data, is_train=True, mappings=None, scaling_params=None):
    """Encodes categorical features and scales numerical features."""
    data_encoded = np.copy(data)
    categorical_features = [name for name, dtype in data.dtype.fields.items()
                            if dtype[0].kind == 'U' and name not in ['city']]
    numerical_features = [name for name, dtype in data.dtype.fields.items()
                          if dtype[0].kind in 'if' and name not in ['enrollee_id', 'training_hours']]

    if is_train:
        mappings = {}
        scaling_params = {}

    # 1. SPECIAL: Encode 'city' with top N frequent cities + 'Other_City'
    top_n_cities = 15
    if 'city' in data.dtype.names:
        if is_train:
            unique_cities, counts = np.unique(data['city'], return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            top_cities = unique_cities[sorted_indices[:top_n_cities]]

            mappings['city'] = {city: i for i, city in enumerate(top_cities)}
            mappings['city']['Other_City'] = len(top_cities)

        city_encoded = []
        for city in data['city']:
            if city in mappings['city']:
                city_encoded.append(mappings['city'][city])
            else:
                city_encoded.append(mappings['city']['Other_City'])

        data_encoded['city'] = np.array(city_encoded)

    # 2. SPECIAL: Handle 'experience' column (<1, >20)
    if 'experience' in data.dtype.names:
        if is_train:
            mappings['experience'] = {'<1': 0, '>20': 21}

        experience_encoded = []
        for val in data['experience']:
            if val == '<1':
                experience_encoded.append(0)
            elif val == '>20':
                experience_encoded.append(21)
            else:
                try:
                    experience_encoded.append(int(val))
                except ValueError:
                    experience_encoded.append(-1)  # Unknown/error value

        data_encoded['experience'] = np.array(experience_encoded)

    # 3. SPECIAL: Handle 'company_size' column (ordinal encoding)
    if 'company_size' in data.dtype.names:
        if is_train:
            mappings['company_size'] = {
                '<10': 0, '10/49': 1, '50-99': 2, '100-499': 3,
                '500-999': 4, '1000-4999': 5, '5000-9999': 6, '10000+': 7,
                'Unknown': -1
            }

        company_size_encoded = np.array([mappings['company_size'].get(val, -1) for val in data['company_size']])
        data_encoded['company_size'] = company_size_encoded

    # 4. SPECIAL: Handle 'last_new_job' column (never, >4)
    if 'last_new_job' in data.dtype.names:
        if is_train:
            mappings['last_new_job'] = {'never': 0, '>4': 5}

        last_new_job_encoded = []
        for val in data['last_new_job']:
            if val == 'never':
                last_new_job_encoded.append(0)
            elif val == '>4':
                last_new_job_encoded.append(5)
            else:
                try:
                    last_new_job_encoded.append(int(val))
                except ValueError:
                    last_new_job_encoded.append(-1)

        data_encoded['last_new_job'] = np.array(last_new_job_encoded)

    # 5. STANDARD: Encode other categorical features (label encoding)
    categorical_features = ['gender', 'relevent_experience', 'enrolled_university',
                            'education_level', 'major_discipline', 'company_type']

    for name in categorical_features:
        if name not in data.dtype.names:
            continue

        if is_train:
            unique_values = np.unique(data[name])
            mappings[name] = {val: i for i, val in enumerate(unique_values)}

        data_encoded[name] = np.array([mappings[name].get(val, -1) for val in data[name]])

    # 6. SPECIAL: Log transformation for 'training_hours'
    if 'training_hours' in data.dtype.names:
        if is_train:
            log_values = np.log1p(data['training_hours'])
            mean = np.mean(log_values)
            std = np.std(log_values)

            scaling_params['training_hours'] = {
                'mean': mean,
                'std': std,
                'method': 'log_transform'
            }
        else:
            mean = scaling_params['training_hours']['mean']
            std = scaling_params['training_hours']['std']

        log_values = np.log1p(data['training_hours'])
        data_encoded['training_hours'] = (log_values - mean) / std

    # 7. STANDARD: Scale 'city_development_index' (Standard Scaling)
    if 'city_development_index' in data.dtype.names:
        if is_train:
            mean = np.mean(data['city_development_index'])
            std = np.std(data['city_development_index'])
            scaling_params['city_development_index'] = {'mean': mean, 'std': std, 'method': 'standard'}
        else:
            mean = scaling_params['city_development_index']['mean']
            std = scaling_params['city_development_index']['std']

        data_encoded['city_development_index'] = (data['city_development_index'] - mean) / std

    return data_encoded, mappings, scaling_params
import os
import subprocess
import sys
import json
import sqlite3

# ==========================================
# 1. Function Reflection Logic (apis.json)
# ==========================================

REFLECTION_FUNC_PHP_CODE = """
<?php
// Define str_starts_with and str_contains for PHP versions < 8.0
if (!function_exists('str_starts_with')) {
    function str_starts_with($haystack, $needle) {
        return strpos($haystack, $needle) === 0;
    }
}

if (!function_exists('str_contains')) {
    function str_contains($haystack, $needle) {
        return $needle === '' || strpos($haystack, $needle) !== false;
    }
}

function collect_functions($vars) {
    // Get all loaded extensions
    $extensions = get_loaded_extensions();

    // Initialize an array to hold all internal and extension functions
    $allInternalFunctions = array();

    // Get all defined functions
    $definedFunctions = get_defined_functions();
    $internalFunctions = $definedFunctions['internal'];
    $allInternalFunctions = array_merge($allInternalFunctions, $internalFunctions);

    // Iterate over each extension to get its functions
    foreach ($extensions as $extension) {
        $functions = get_extension_funcs($extension);
        if ($functions !== false) {
            $allInternalFunctions = array_merge($allInternalFunctions, $functions);
        }
    }

    // Remove duplicates
    $allInternalFunctions = array_unique($allInternalFunctions);

    // Define the skipFunction
    function skipFunction($function): bool {
        if (false
            /* expect input / hang */
         || $function === 'readline'
         || $function === 'readline_read_history'
         || $function === 'readline_write_history'
            /* terminates script */
         || $function === 'exit'
         || $function === 'die'
            /* intentionally violate invariants */
         || $function === 'zend_create_unterminated_string'
         || $function === 'zend_test_array_return'
         || $function === 'zend_test_crash'
         || $function === 'zend_leak_bytes'
            /* mess with output */
         || (is_string($function) && str_starts_with($function, 'ob_'))
         || $function === 'output_add_rewrite_var'
         || $function === 'error_log'
            /* may spend a lot of time waiting for connection timeouts */
         || (is_string($function) && str_contains($function, 'connect'))
         || (is_string($function) && str_starts_with($function, 'snmp'))
            /* misc */
         || $function === 'mail'
         || $function === 'mb_send_mail'
         || $function === 'pcntl_fork'
         || $function === 'pcntl_rfork'
         || $function === 'posix_kill'
         || $function === 'posix_setrlimit'
         || $function === 'sapi_windows_generate_ctrl_event'
         || $function === 'imagegrabscreen'
         || $function === 'zend_delref' # we should exclude zend_delref, see #18242 and #18848
        ) {
            return false;
        }

        // These conditions won't be true in this context but are included for completeness
        if (is_array($function) && get_class($function[0]) === mysqli::class
            && in_array($function[1], ['__construct', 'connect', 'real_connect'])) {
            return false;
        }

        if (is_array($function) && $function[0] instanceof SoapServer) {
            /* TODO: Uses fatal errors */
            return false;
        }

        return true;
    }

    // Filter functions using skipFunction
    $allInternalFunctions = array_filter($allInternalFunctions, 'skipFunction');

    // Sort the functions
    sort($allInternalFunctions);

    $functionInfoList = [];

    foreach ($allInternalFunctions as $functionName) {
        try {
            // Get reflection of the function to determine the parameters
            $reflection = new ReflectionFunction($functionName);
            $numParams = $reflection->getNumberOfParameters();
            $params = $reflection->getParameters();

            // Prepare parameter info
            $paramInfos = [];
            foreach ($params as $param) {
                $paramDetails = [
                    'name' => $param->getName(),
                    'type' => $param->hasType() ? (string)$param->getType() : null,
                    'is_optional' => $param->isOptional(),
                    'default_value' => null,
                ];

                // Suppress deprecation warnings when getting default value
                if ($param->isDefaultValueAvailable()) {
                    $originalErrorReporting = error_reporting();
                    error_reporting($originalErrorReporting & ~E_DEPRECATED);
                    $defaultValue = $param->getDefaultValue();
                    error_reporting($originalErrorReporting);

                    // Convert default value to a JSON-serializable format
                    if (is_scalar($defaultValue) || is_null($defaultValue)) {
                        $paramDetails['default_value'] = $defaultValue;
                    } else {
                        // Convert non-scalar values to their string representation
                        $paramDetails['default_value'] = var_export($defaultValue, true);
                    }
                }

                $paramInfos[] = $paramDetails;
            }

            // Collect function info
            $functionInfo = [
                'name' => $functionName,
                'num_params' => $numParams,
                'params' => $paramInfos,
            ];

            $functionInfoList[] = $functionInfo;

        } catch (\Throwable $e) {
            // Handle any exceptions or errors
            // You can log the error if needed
        }
    }

    // Write the function info list to JSON file
    $json = json_encode($functionInfoList, JSON_PRETTY_PRINT);

    // Check if json_encode failed
    if ($json === false) {
        echo "json_encode error: " . json_last_error_msg() . "\n";
        // Optionally, you can handle the error further here
    } else {
        file_put_contents('./apis.json', $json);
    }
}

// Call the function
collect_functions([]);
"""

# ==========================================
# 2. Class Reflection Logic (class.json)
# ==========================================

REFLECTION_CLASS_PHP_CODE = """
<?php

function skipFunction($function): bool {
    if (false
        /* expect input / hang */
     || $function === 'readline'
     || $function === 'readline_read_history'
     || $function === 'readline_write_history'
        /* terminates script */
     || $function === 'exit'
     || $function === 'die'
        /* intentionally violate invariants */
     || $function === 'zend_create_unterminated_string'
     || $function === 'zend_test_array_return'
     || $function === 'zend_test_crash'
     || $function === 'zend_leak_bytes'
        /* mess with output */
     || (is_string($function) && str_starts_with($function, 'ob_'))
     || $function === 'output_add_rewrite_var'
     || $function === 'error_log'
        /* may spend a lot of time waiting for connection timeouts */
     || (is_string($function) && str_contains($function, 'connect'))
     || (is_string($function) && str_starts_with($function, 'snmp'))
     || (is_array($function) && isset($function[0]) && is_object($function[0]) && get_class($function[0]) === mysqli::class
         && in_array($function[1], ['__construct', 'connect', 'real_connect']))
        /* misc */
     || $function === 'mail'
     || $function === 'mb_send_mail'
     || $function === 'pcntl_fork'
     || $function === 'pcntl_rfork'
     || $function === 'posix_kill'
     || $function === 'posix_setrlimit'
     || $function === 'sapi_windows_generate_ctrl_event'
     || $function === 'imagegrabscreen'
    ) {
        return true;
    }
    if (is_array($function) && isset($function[0]) && is_object($function[0]) && $function[0] instanceof SoapServer) {
        /* TODO: Uses fatal errors */
        return true;
    }

    return false;
}

// Main code to collect info from all classes

// Get all declared classes
$classes = get_declared_classes();
if (empty($classes)) {
    echo "No classes declared.\n";
    exit;
}

// Prepare array to hold info for all classes
$allClassesInfo = [];

foreach ($classes as $className) {

    // Skip classes that cannot be instantiated without constructor
    $rc = new ReflectionClass($className);
    if ($rc->isAbstract() || $rc->isInterface() || ($rc->isInternal() && $rc->isFinal())) {
        continue;
    }

    $classInfo = [];
    $classInfo['class_name'] = $className;

    // Collect the class attributes (properties) names
    $properties = $rc->getProperties();
    $propertyNames = [];
    foreach ($properties as $property) {
        $propertyNames[] = $property->getName();
    }
    $classInfo['attributes'] = $propertyNames;

    // Get all methods of the class
    $methods = $rc->getMethods();
    $classInfo['methods'] = [];

    if (!empty($methods)) {
        // Collect method information
        foreach ($methods as $method) {
            $methodName = $method->getName();

            // Skip methods that should be skipped
            if (skipFunction([$className, $methodName])) {
                continue;
            }

            if ($method->isAbstract() || !$method->isPublic()) {
                // Skip abstract or non-public methods
                continue;
            }

            // Collect method info
            $methodInfo = [];
            $methodInfo['name'] = $methodName;

            // Collect parameter count
            $parameters = $method->getParameters();
            $methodInfo['params_count'] = count($parameters);

            $classInfo['methods'][] = $methodInfo;
        }
    }

    $allClassesInfo[] = $classInfo;
}

// Dump all class info into class.json
file_put_contents('class.json', json_encode($allClassesInfo, JSON_PRETTY_PRINT));
"""

# ==========================================
# 3. Python Data Loading Helpers
# ==========================================

def load_json_file(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: {json_file} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_file}.")
        return []

# --- Functions DB Helpers ---

def create_func_database(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS functions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            num_params INTEGER NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            function_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            type TEXT,
            is_optional INTEGER NOT NULL,
            default_value TEXT,
            FOREIGN KEY (function_id) REFERENCES functions (id)
        )
    ''')
    conn.commit()
    return conn

def insert_func_data(conn, data):
    cursor = conn.cursor()
    for function in data:
        function_name = function.get('name')
        num_params = function.get('num_params', 0)
        params = function.get('params', [])

        cursor.execute('INSERT INTO functions (name, num_params) VALUES (?, ?)', 
                      (function_name, num_params))
        function_id = cursor.lastrowid

        for param in params:
            param_name = param.get('name')
            param_type = param.get('type')
            is_optional = 1 if param.get('is_optional') else 0
            default_value = param.get('default_value')
            if default_value is not None:
                default_value = str(default_value)

            cursor.execute('''
                INSERT INTO parameters (function_id, name, type, is_optional, default_value)
                VALUES (?, ?, ?, ?, ?)
            ''', (function_id, param_name, param_type, is_optional, default_value))
    conn.commit()

# --- Classes DB Helpers ---

def create_class_database(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT UNIQUE
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attributes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER,
            name TEXT,
            FOREIGN KEY (class_id) REFERENCES classes (id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS methods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER,
            name TEXT,
            params_count INTEGER,
            FOREIGN KEY (class_id) REFERENCES classes (id)
        )
    ''')
    conn.commit()
    return conn

def insert_class_data(conn, data):
    cursor = conn.cursor()
    for class_info in data:
        class_name = class_info['class_name']
        
        cursor.execute('INSERT OR IGNORE INTO classes (class_name) VALUES (?)', (class_name,))
        
        # Get class_id (whether inserted or ignored)
        cursor.execute('SELECT id FROM classes WHERE class_name = ?', (class_name,))
        row = cursor.fetchone()
        if not row:
            continue
        class_id = row[0]

        # Insert attributes
        for attr_name in class_info.get('attributes', []):
            cursor.execute('INSERT INTO attributes (class_id, name) VALUES (?, ?)', (class_id, attr_name))

        # Insert methods
        for method_info in class_info.get('methods', []):
            method_name = method_info['name']
            params_count = method_info['params_count']
            cursor.execute('INSERT INTO methods (class_id, name, params_count) VALUES (?, ?, ?)', 
                          (class_id, method_name, params_count))
    conn.commit()

# ==========================================
# 4. Main Reflect Controller
# ==========================================

def run_php_script(project_root, script_name, php_code):
    """Writes PHP code to file, executes it, returns success status."""
    php_binary = os.path.join(project_root, "php-src", "sapi", "cli", "php")
    script_path = os.path.join(project_root, script_name)

    if not os.path.exists(php_binary):
        print(f"Error: PHP binary not found at {php_binary}.")
        return False

    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(php_code)
    except OSError as e:
        print(f"Error writing {script_name}: {e}")
        return False

    try:
        subprocess.run([php_binary, script_name], cwd=project_root, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_name}: {e}")
        return False
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

def reflect(project_root):
    """
    Executes PHP reflection logic for both Functions and Classes,
    populating apis.db and class.db respectively.
    """
    print(f"Starting PHP Reflection in {project_root}...")

    # --- Part A: Functions (apis.json -> apis.db) ---
    if run_php_script(project_root, "reflect_funcs.php", REFLECTION_FUNC_PHP_CODE):
        json_path = os.path.join(project_root, "apis.json")
        db_path = os.path.join(project_root, "apis.db")
        
        if os.path.exists(json_path):
            print(f"Importing {json_path} into {db_path}...")
            data = load_json_file(json_path)
            if data:
                conn = create_func_database(db_path)
                insert_func_data(conn, data)
                conn.close()
                print("Functions DB created successfully.")
            else:
                print("Warning: apis.json was empty.")
        else:
            print("Error: apis.json not generated.")

    # --- Part B: Classes (class.json -> class.db) ---
    if run_php_script(project_root, "reflect_classes.php", REFLECTION_CLASS_PHP_CODE):
        json_path = os.path.join(project_root, "class.json")
        db_path = os.path.join(project_root, "class.db")

        if os.path.exists(json_path):
            print(f"Importing {json_path} into {db_path}...")
            data = load_json_file(json_path)
            if data:
                conn = create_class_database(db_path)
                insert_class_data(conn, data)
                conn.close()
                print("Classes DB created successfully.")
            else:
                print("Warning: class.json was empty.")
        else:
            print("Error: class.json not generated.")
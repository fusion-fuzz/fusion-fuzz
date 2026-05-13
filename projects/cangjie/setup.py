import os
import shutil
import subprocess
import sys
import tarfile
import textwrap


# Where the SDK is installed (must match PATH in Dockerfile)
SDK_DIR = "/opt/cangjie"
CJC_BIN = f"{SDK_DIR}/bin/cjc"


def _install_sdk(project_root: str):
    """
    Download (or copy) the Cangjie SDK tarball and extract it to SDK_DIR.

    Resolution order:
    1. CANGJIE_SDK_URL env var  — wget from that URL
    2. Local tarball at projects/cangjie/cangjie-sdk-linux-x64.tar.gz
    """
    # Already installed?
    if os.path.exists(CJC_BIN):
        print(f"Cangjie SDK already installed at {SDK_DIR}.")
        return

    tarball = "/tmp/cangjie-sdk.tar.gz"
    sdk_url = os.environ.get("CANGJIE_SDK_URL", "")
    local_tarball = os.path.join(project_root, "cangjie-sdk-linux-x64.tar.gz")

    if sdk_url:
        print(f"Downloading Cangjie SDK from {sdk_url} ...")
        subprocess.run(
            ["wget", "-q", "--show-progress", sdk_url, "-O", tarball],
            check=True,
        )
    elif os.path.exists(local_tarball):
        print(f"Using local SDK tarball: {local_tarball}")
        tarball = local_tarball
    else:
        print(
            "ERROR: Cangjie SDK not found.\n"
            "Provide one of:\n"
            "  • CANGJIE_SDK_URL=<url>  (env var pointing to cangjie-sdk-linux-x64-VERSION.tar.gz)\n"
            "  • projects/cangjie/cangjie-sdk-linux-x64.tar.gz  (pre-downloaded tarball)\n"
            "Download from: https://cangjie-lang.cn/en/download"
        )
        sys.exit(1)

    print(f"Extracting SDK to {SDK_DIR} ...")
    os.makedirs(SDK_DIR, exist_ok=True)
    with tarfile.open(tarball) as tf:
        # Strip the top-level 'cangjie/' directory if present
        members = tf.getmembers()
        prefix = members[0].name.split("/")[0] + "/" if members else ""
        for member in members:
            if member.name.startswith(prefix):
                member.name = member.name[len(prefix):]
            if member.name:
                tf.extract(member, SDK_DIR)

    if tarball == "/tmp/cangjie-sdk.tar.gz":
        os.remove(tarball)

    if not os.path.exists(CJC_BIN):
        print(f"ERROR: extraction succeeded but {CJC_BIN} not found. Check the SDK layout.")
        sys.exit(1)

    print("SDK extraction complete.")


def setup(project_root):
    """
    Sets up the Cangjie fuzzing environment (runs inside the ffl-cangjie container):
    1. Downloads and installs the Cangjie SDK (cjc) if not already present.
    2. Clones open-source Cangjie example repos to harvest .cj seed files.
    3. Generates a small set of hand-written synthetic seeds.
    4. Copies all seeds into projects/cangjie/seeds/.
    """
    print(f"Setting up Cangjie in: {project_root}")

    def _run(cmd_str, cwd=None, check=True):
        print(f"[run] {cmd_str[:100]}...")
        subprocess.run(["sh", "-c", cmd_str], check=check, cwd=cwd)

    # 1. Install SDK
    _install_sdk(project_root)

    result = subprocess.run([CJC_BIN, "-v"], capture_output=True, text=True)
    print(f"cjc available: {(result.stdout + result.stderr).strip()}")

    seeds_dir = os.path.join(project_root, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    # 2. Clone community repos for seed harvesting
    repos = [
        ("https://github.com/open-cangjie/hello-cangjie.git",      "hello-cangjie"),
        ("https://github.com/waylau/cangjie-programming-language-tutorial.git", "cj-tutorial"),
        ("https://github.com/open-cangjie/cangjie-examples.git",    "cangjie-examples"),
    ]

    tmp_dir = os.path.join(project_root, "_seed_repos")
    os.makedirs(tmp_dir, exist_ok=True)

    for url, name in repos:
        dest = os.path.join(tmp_dir, name)
        if os.path.exists(dest):
            print(f"Repo {name} already cloned.")
        else:
            print(f"Cloning {url} ...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth=1", url, dest],
                    check=True,
                    timeout=120,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"Warning: could not clone {url}: {e}. Skipping.")

    # Collect all .cj files from cloned repos
    collected = skipped = 0
    for _, name in repos:
        repo_dir = os.path.join(tmp_dir, name)
        if not os.path.exists(repo_dir):
            continue
        for root, _, files in os.walk(repo_dir):
            for fname in files:
                if not fname.endswith(".cj"):
                    continue
                src = os.path.join(root, fname)
                rel = os.path.relpath(src, repo_dir)
                safe_name = f"{name}__{rel.replace(os.sep, '__')}"
                dst = os.path.join(seeds_dir, safe_name)
                if os.path.exists(dst):
                    skipped += 1
                    continue
                try:
                    shutil.copy2(src, dst)
                    collected += 1
                except Exception as e:
                    print(f"Warning: could not copy {src}: {e}")

    print(f"Collected {collected} .cj seeds from repos ({skipped} already existed).")

    # 3. Write synthetic seed files covering key language features
    _write_synthetic_seeds(seeds_dir)

    total = len([f for f in os.listdir(seeds_dir) if f.endswith(".cj")])
    print(f"Total seeds in {seeds_dir}: {total}")
    print("Cangjie setup complete.")


def _write_synthetic_seeds(seeds_dir: str):
    seeds = {
        "syn_hello.cj": textwrap.dedent("""\
            main() {
                println("Hello, Cangjie!")
            }
            """),

        "syn_variables.cj": textwrap.dedent("""\
            main() {
                let x: Int64 = 42
                var y: Float64 = 3.14
                let name: String = "cangjie"
                println(x)
                println(y)
                println(name)
            }
            """),

        "syn_func.cj": textwrap.dedent("""\
            func add(a: Int64, b: Int64): Int64 {
                return a + b
            }

            func greet(name: String): String {
                return "Hello, " + name + "!"
            }

            main() {
                println(add(1, 2))
                println(greet("world"))
            }
            """),

        "syn_class.cj": textwrap.dedent("""\
            open class Animal {
                var name: String
                var age: Int64

                init(name: String, age: Int64) {
                    this.name = name
                    this.age = age
                }

                public open func speak(): String {
                    return name + " says hello"
                }
            }

            class Dog <: Animal {
                init(name: String, age: Int64) {
                    super(name, age)
                }

                public override func speak(): String {
                    return name + " barks"
                }
            }

            main() {
                let d = Dog("Rex", 3)
                println(d.speak())
            }
            """),

        "syn_enum.cj": textwrap.dedent("""\
            enum Direction {
                | North
                | South
                | East
                | West
            }

            func describe(d: Direction): String {
                match (d) {
                    case North => "up"
                    case South => "down"
                    case East  => "right"
                    case West  => "left"
                }
            }

            main() {
                println(describe(Direction.North))
            }
            """),

        "syn_generics.cj": textwrap.dedent("""\
            func identity<T>(x: T): T {
                return x
            }

            struct Pair<A, B> {
                var first: A
                var second: B

                init(first: A, second: B) {
                    this.first = first
                    this.second = second
                }
            }

            main() {
                println(identity(100))
                println(identity("hello"))
                let p = Pair<Int64, String>(1, "one")
                println(p.first)
            }
            """),

        "syn_loops.cj": textwrap.dedent("""\
            main() {
                var sum: Int64 = 0
                for (i in 0..10) {
                    sum = sum + i
                }
                println(sum)

                var n: Int64 = 1
                while (n < 100) {
                    n = n * 2
                }
                println(n)

                var x: Int64 = 0
                do {
                    x = x + 1
                } while (x < 5)
                println(x)
            }
            """),

        "syn_lambda.cj": textwrap.dedent("""\
            main() {
                let double = { x: Int64 => x * 2 }
                let add = { a: Int64, b: Int64 => a + b }
                println(double(21))
                println(add(3, 4))
            }
            """),

        "syn_interface.cj": textwrap.dedent("""\
            interface Shape {
                func area(): Float64
                func perimeter(): Float64
            }

            class Circle <: Shape {
                var radius: Float64

                init(radius: Float64) {
                    this.radius = radius
                }

                public func area(): Float64 {
                    return 3.14159 * radius * radius
                }

                public func perimeter(): Float64 {
                    return 2.0 * 3.14159 * radius
                }
            }

            main() {
                let c: Shape = Circle(5.0)
                println(c.area())
                println(c.perimeter())
            }
            """),

        "syn_option.cj": textwrap.dedent("""\
            func safeDivide(a: Int64, b: Int64): Option<Int64> {
                if (b == 0) {
                    return None
                }
                return Some(a / b)
            }

            main() {
                match (safeDivide(10, 2)) {
                    case Some(v) => println(v)
                    case None    => println("division by zero")
                }
                match (safeDivide(10, 0)) {
                    case Some(v) => println(v)
                    case None    => println("division by zero")
                }
            }
            """),

        "syn_string_ops.cj": textwrap.dedent("""\
            main() {
                let s: String = "Hello, Cangjie!"
                println(s.size)
                let parts = s.split(", ")
                for (part in parts) {
                    println(part)
                }
                let greeting = "Hello" + ", " + "world!"
                println(greeting)
            }
            """),

        "syn_array.cj": textwrap.dedent("""\
            main() {
                var arr: Array<Int64> = [1, 2, 3, 4, 5]
                for (x in arr) {
                    println(x)
                }
                arr[0] = 10
                println(arr[0])
                println(arr.size)
            }
            """),

        "syn_exception.cj": textwrap.dedent("""\
            class MyError <: Exception {
                init(msg: String) {
                    super(msg)
                }
            }

            func riskyOp(x: Int64): Int64 {
                if (x < 0) {
                    throw MyError("negative input")
                }
                return x * x
            }

            main() {
                try {
                    println(riskyOp(5))
                    println(riskyOp(-1))
                } catch (e: MyError) {
                    println("caught: " + e.message)
                }
            }
            """),
    }

    written = 0
    for filename, content in seeds.items():
        dst = os.path.join(seeds_dir, filename)
        if not os.path.exists(dst):
            with open(dst, "w", encoding="utf-8") as f:
                f.write(content)
            written += 1

    print(f"Wrote {written} synthetic .cj seeds.")

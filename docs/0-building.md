# Building Clobber

Scripts are provided in the [bootstrap](../bootstrap/) directory that installs all the required dependencies.
The [`bootstrap.py`](../bootstrap/bootstrap.py) script bootstraps `vcpkg` and `llvm` together, generating a `CMakePresets.json`.

Bootstrap requirements:

- python > 3.8
- cmake
- git

<br/>
Simply call the main bootstrap script in the **base** directory.
The script checks whether `vcpkg` exists as an executable, otherwise, it will bootstrap and use a local copy.
From there, it will initialize all the dependencies from `vcpkg`, and then build `llvm` from scratch.

```pwsh
py bootstrap/bootstrap.py
```

## Linking an existing LLVM build

TODO

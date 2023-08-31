# libmapper + MediaPipe: Hands

This repo presents a proper binding between libmapper and MediaPipe: Hands.

![Screenshot from 2023-07-31 16-30-02](https://github.com/peacheym/mp-hands-libmapper/assets/15327742/2cb06fcf-7aea-44b7-887c-7d510e4fdf92)

## Installation

To use the binding scripts found in this repo, you'll need to install the following PIP packages.

- [Mediapipe](https://pypi.org/project/mediapipe/) | `pip install mediapipe`
- [libmapper](https://pypi.org/project/libmapper/) | `pip install libmapper`

You can then either clone the git repository in its entirety or download the `mph-bindings.py` script.

## Inatilizing the Bindings

To initalize the bindings, run the binding script:

```bash
./mph-bindings.py
```

You can add flags to change the behavior of the script. For example, `./mph-bindings.py --max-hands X`, where `X` is an integer will set the amount of hands that mediapipe will track at once.

```bash
./mph-bindings.py --help
```

will provide more information about the available flags.

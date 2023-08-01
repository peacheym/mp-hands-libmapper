# libmapper + MediaPipe: Hands

This repo presents a proper binding between libmapper and MediaPipe: Hands.

![mph_v](https://github.com/peacheym/mp-hands-libmapper/assets/15327742/e963c1a1-f4f2-438c-93fc-e29749e0eb43)


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

You can add flags to change the behaviour of the script. For example, `./mph-bindings.py --max-hands X`, where `X` is an integer will set the amount of hands that mediapipe will track at once.

```bash
./mph-bindings.py --help
```

will provide more information about the available flags.

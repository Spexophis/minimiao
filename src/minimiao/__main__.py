from . import main

def app():
    app_wrapper = main.AppWrapper()
    app_wrapper.run()


if __name__ == "__main__":
    raise SystemExit(app())

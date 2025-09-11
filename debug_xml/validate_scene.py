import sys
import mujoco

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_scene.py <path-to-xml>")
        sys.exit(1)
    path = sys.argv[1]
    try:
        mujoco.MjModel.from_xml_path(path)
        print("OK:", path)
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    main()


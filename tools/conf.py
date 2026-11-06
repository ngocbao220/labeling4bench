"""Đọc config/ bằng Hydra (compose API) cho các tool dùng argparse. Trả về dict thường."""
import os

CONF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

# ${repo:} trong yaml = thư mục gốc repo, để log Hydra không phụ thuộc thư mục đang đứng khi gõ lệnh
from omegaconf import OmegaConf as _OC  # noqa: E402
if not _OC.has_resolver("repo"):
    _OC.register_new_resolver("repo", lambda: os.path.dirname(CONF_DIR))


def path(cfg, key):
    """Đường dẫn trong cfg['paths'] → tuyệt đối (tương đối thì tính từ gốc repo)."""
    p = cfg["paths"][key]
    return p if os.path.isabs(p) else os.path.join(os.path.dirname(CONF_DIR), p)


def load(overrides=()):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose("config", overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def split_argv(argv=None):
    """Tách ghi đè Hydra (key=value, không bắt đầu bằng '-') khỏi tham số argparse, để đặt ở đâu cũng được."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    ov = [x for x in argv if "=" in x and not x.startswith("-")]
    return [x for x in argv if x not in ov], ov


if __name__ == "__main__":
    # In lệnh `vllm serve` cho các mô hình có base_url, lấy path/served_name/cổng từ config:
    #   python3 tools/conf.py models=local
    import sys
    from urllib.parse import urlparse
    _, ov = split_argv()
    for k, m in load(ov)["models"].items():
        if m.get("base_url") and m.get("path"):
            print(f"# {k}\nvllm serve {m['path']} --served-model-name {m['served_name']} --port {urlparse(m['base_url']).port or 8000} "
                  + " ".join(m.get("serve_args", [])))

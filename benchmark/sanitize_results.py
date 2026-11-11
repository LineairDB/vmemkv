#!/usr/bin/env python3
"""Remove machine and cloud-resource identifiers from benchmark JSON artifacts."""

import argparse
import json
from pathlib import Path

PRIVATE_KEYS = {
    "ami_id",
    "aws_account_id",
    "account_id",
    "host_name",
    "hostname",
    "local_hostname",
    "instance_id",
    "public_ip",
    "private_ip",
    "public_dns_name",
    "private_dns_name",
    "security_group_id",
    "key_name",
}


def sanitize_public_metadata(value):
    if isinstance(value, dict):
        return {
            key: sanitize_public_metadata(item)
            for key, item in value.items()
            if key.lower() not in PRIVATE_KEYS
        }
    if isinstance(value, list):
        return [sanitize_public_metadata(item) for item in value]
    return value


def sanitize_file(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cleaned = sanitize_public_metadata(data)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_files", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.json_files:
        sanitize_file(path)


if __name__ == "__main__":
    main()

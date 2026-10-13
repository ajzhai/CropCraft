#!/usr/bin/env python3
"""Send a JSON POST to webhook URL stored in ~/.config/ggssvt/webhook.conf

Usage:
  echo '{"msg":"hi"}' | python tools/notify_webhook.py
  python tools/notify_webhook.py --body-file /path/to/file

The config file must contain the webhook URL on the first line.
"""
import sys
import os
import argparse
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

CFG = os.path.expanduser('~/.config/ggssvt/webhook.conf')


def read_config(path=CFG):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Webhook config not found: {path}")
    url = open(path, 'r', encoding='utf-8').read().strip().splitlines()[0].strip()
    if not url:
        raise ValueError('webhook.conf contains no URL')
    return url


def post_json(url: str, data: bytes):
    req = Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=15) as resp:
            status = resp.getcode()
            body = resp.read().decode('utf-8', errors='ignore')
            print(f"Webhook POST {status}: {body}")
            return 0
    except HTTPError as e:
        print(f"HTTPError {e.code}: {e.reason}", file=sys.stderr)
        return 2
    except URLError as e:
        print(f"URLError: {e.reason}", file=sys.stderr)
        return 3


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--body-file', help='Read body from file instead of stdin')
    args = p.parse_args(argv)
    if args.body_file:
        body = open(args.body_file, 'r', encoding='utf-8').read()
    else:
        body = sys.stdin.read()
    body = body.strip()
    try:
        # If the body looks like JSON, use it; otherwise wrap it
        if body.startswith('{') or body.startswith('['):
            payload = json.loads(body)
        else:
            payload = {'message': body}
    except Exception:
        payload = {'message': body}
    data = json.dumps(payload).encode('utf-8')
    url = read_config()
    rc = post_json(url, data)
    sys.exit(rc)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Send a simple email using SMTP config from ~/.config/ggssvt/notify.conf

Usage:
  python tools/notify_email.py --subject "subj" --body-file /path/to/body.txt
  or
  echo "body" | python tools/notify_email.py --subject "subj"

The config file must contain a single CSV line:
server,port,use_tls(yes/no),username,password,from_email,to_email1;to_email2
"""
import sys
import os
import argparse
import smtplib
from email.message import EmailMessage

CFG = os.path.expanduser('~/.config/ggssvt/notify.conf')


def read_config(path=CFG):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Notify config not found: {path}")
    line = open(path, 'r', encoding='utf-8').read().strip()
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 7:
        raise ValueError('notify.conf must have 7 comma-separated values')
    server, port, use_tls, username, password, from_email, to_emails = parts[:7]
    to_emails = [e.strip() for e in to_emails.split(';') if e.strip()]
    return server, int(port), use_tls.lower() in ('yes', 'true', '1'), username, password, from_email, to_emails


def send_email(subject: str, body: str):
    server, port, use_tls, username, password, from_email, to_emails = read_config()
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = ', '.join(to_emails)
    msg.set_content(body)

    if use_tls:
        with smtplib.SMTP(server, port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(username, password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(server, port, timeout=30) as smtp:
            smtp.send_message(msg)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--subject', required=True)
    p.add_argument('--body-file', help='Path to a file whose content is the body. If omitted, read stdin')
    args = p.parse_args(argv)
    if args.body_file:
        body = open(args.body_file, 'r', encoding='utf-8').read()
    else:
        body = sys.stdin.read()
    send_email(args.subject, body)


if __name__ == '__main__':
    main()

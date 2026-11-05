import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import jsonschema
import pytest

import corread
from corread import TrainingInstance, paths, reporting

from .conftest import TINY

SCHEMA = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'docs', 'reporting.schema.json')))
TOKEN = 'test-token-not-a-secret'


class Server:
    """A local HTTP endpoint that records what it receives and answers with ``status``."""

    def __init__(self, status=200, redirect=None):
        received = self.received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                received.append({'headers': dict(self.headers), 'body': json.loads(body)})
                self.send_response(302 if redirect else status)
                if redirect:
                    self.send_header('Location', redirect)
                self.send_header('Content-Length', '2')
                self.end_headers()
                self.wfile.write(b'no')

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(('127.0.0.1', 0), Handler)
        self.url = 'http://127.0.0.1:%d/api/runs' % self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


def configure(monkeypatch, root, url, every=1000):
    config = os.path.join(root, 'paths.toml')
    folders = open(config).read().split('\n[report]')[0]
    with open(config, 'w') as f:
        f.write(folders + '\n[report]\nurl = "%s"\nevery = %s\n' % (url, every))
    monkeypatch.setenv(reporting.TOKEN_VAR, TOKEN)
    paths.clear_cache()


def test_settings_are_checked_up_front(root, monkeypatch):
    with pytest.raises(ValueError, match=r'\[report\]'):
        reporting.settings()
    configure(monkeypatch, root, 'http://example.org/api')
    with pytest.raises(ValueError, match='https'):
        reporting.settings()
    configure(monkeypatch, root, 'https://example.org/api')
    monkeypatch.delenv(reporting.TOKEN_VAR)
    with pytest.raises(ValueError, match=reporting.TOKEN_VAR):
        TrainingInstance(report=True, **TINY)


def test_wordset_reports(root, monkeypatch, server):
    configure(monkeypatch, root, server.url)
    corread.generate_training_set(
        'ws',
        'words',
        train_fonts=['sans'],
        val_fonts=['sans'],
        num_train=2,
        num_val=1,
        width=100,
        height=100,
        seed=1,
        report=True,
    )
    assert server.received, 'no report arrived'
    last = server.received[-1]
    assert last['headers']['Authorization'] == 'Bearer ' + TOKEN
    assert last['headers']['User-Agent'] == 'CORread-reporter/2'
    assert last['body']['kind'] == 'wordset' and last['body']['status'] == 'complete'
    for report in server.received:
        jsonschema.validate(report['body'], SCHEMA)
    corread.report_wordset_status('ws')
    assert server.received[-1]['body']['event'] == 'snapshot'


def test_training_and_lesion_payloads_follow_the_schema(in_trained):
    models = os.path.join(in_trained, 'models')
    for group, name in (
        ('pre', 'pre_z_1'),
        ('illiterate', 'illit_z_3'),
        ('words', 'lit_bias_z_3'),
        ('words', 'lit_no_bias_z_3'),
    ):
        meta = json.load(open(os.path.join(models, group, name, 'metadata.json')))
        for status, live in (('in progress', ('training', 3, 10)), ('complete', None), ('stopped', None)):
            payload = reporting.training_payload(meta, group, 'snapshot', status, live)
            jsonschema.validate(payload, SCHEMA)
        assert payload['config']['model_class'] == meta['model']['class']
    lesions = [n for n in os.listdir(os.path.join(models, 'words')) if n.startswith('lesion_')]
    for name in lesions:
        meta = json.load(open(os.path.join(models, 'words', name, 'metadata.json')))
        jsonschema.validate(reporting.lesion_payload(meta, 'words', 'snapshot', 'complete'), SCHEMA)


def test_failed_report_warns_once_without_the_token(root, monkeypatch, caplog):
    rejecting = Server(status=401)
    try:
        reporter = reporting.Reporter(rejecting.url, TOKEN, every=1000)
        with caplog.at_level(logging.WARNING, logger='corread.reporting'):
            reporter.send({'n': 1})
            reporter.close()
        assert len(caplog.records) == 1 and 'HTTP 401' in caplog.text
        assert TOKEN not in caplog.text
    finally:
        rejecting.close()


def test_redirects_are_not_followed(server):
    redirecting = Server(redirect=server.url)
    try:
        with pytest.raises(RuntimeError, match='redirects are not followed'):
            reporting.post(redirecting.url, TOKEN, {'n': 1})
        assert server.received == []  # the token never reached the redirect's target
    finally:
        redirecting.close()

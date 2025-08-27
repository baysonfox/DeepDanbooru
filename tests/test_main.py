#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
from unittest import mock

import numpy
import pytest
import six
from click.testing import CliRunner
from PIL import Image


def test_import():
    import deepdanbooru


@pytest.mark.parametrize('func_name', ['main', 'evaluate'])
def test_help(func_name):
    mod = importlib.import_module('deepdanbooru.__main__')
    runner = CliRunner()
    result = runner.invoke(getattr(mod, func_name), ['--help'])
    assert result.exit_code == 0
    assert result.output


@pytest.fixture
def packages():
    with open('requirements.txt') as f:
        return f.read().splitlines()


def test_package_setup(packages):
    import setuptools
    with mock.patch('setuptools.setup'):
        import setup
        setup_pkgs = setup.install_requires
        pytorch_pkg = setup.pytorch_pkg
    assert setup_pkgs == list(
        filter(lambda x: not x.startswith('torch'), packages))
    assert list(
        filter(lambda x: x.startswith('torch'), packages)
    ) == pytorch_pkg


def test_readme_pkg(packages):
    with open('README.md') as f:
        text = f.read()
    line_start = 'Following packages are need to be installed.'
    line_end = '\n\n'
    pkg_text = text.split(line_start)[1].rsplit(line_end)[0].strip()
    readme_pkgs = list(map(lambda x: x.split(' ', 1)[1], pkg_text.splitlines()))

    assert list(
        sorted(filter(lambda x: not x.startswith('torch'), readme_pkgs))
    ) == list(
        sorted(filter(lambda x: not x.startswith('torch'), packages))
    )

    assert list(
        filter(lambda x: x.startswith('torch'), packages)
    ) == list(
        filter(lambda x: x.startswith('torch'), readme_pkgs)
    )


@pytest.mark.parametrize('use_bytes_io', [True, False])
def test_load_image_for_evaluate(tmp_path, use_bytes_io):
    image_path = tmp_path / 'test.jpg'
    image = Image.new('RGB', (300, 300), color='red')
    image.save(image_path)
    from deepdanbooru.data import load_image_for_evaluate
    if not use_bytes_io:
        res = load_image_for_evaluate(image_path.as_posix(), 299, 299)
    else:
        with open(image_path.as_posix(), 'rb') as f:
            image_input = six.BytesIO(f.read())
        res = load_image_for_evaluate(image_input, 299, 299)
    assert isinstance(res, numpy.ndarray)
    assert res.shape == (299, 299, 3)

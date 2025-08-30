#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for pkg."""

import unirobot


def test_pkg() -> None:
    """Unit test for pkg."""
    assert unirobot.__package__ == "unirobot"

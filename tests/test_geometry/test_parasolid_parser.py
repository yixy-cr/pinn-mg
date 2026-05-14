import pytest
import sys
sys.path.insert(0, 'src/geometry')
from parasolid_parser import ParasolidParser


def test_parser_init():
    """Test parser initialization"""
    parser = ParasolidParser()
    assert parser.header == {}
    assert parser.boundary_points == []


def test_parse_header():
    """Test parsing header information"""
    parser = ParasolidParser()
    sample_data = """**ABCDEFGHIJKLMNOPQRSTUVWXYZ**
**PART1;
KEY=test;
FILE=D:\\test.x_t;
DATE=13-may-2026;
**PART2;
**END_OF_HEADER*****************************************************************
"""
    result = parser.parse_header(sample_data)
    assert result['KEY'] == 'test'
    assert result['FILE'] == 'D:\\test.x_t'


def test_load_file():
    """Test loading .x_t file"""
    parser = ParasolidParser()
    # Use existing file
    filepath = '1pinflue4wire.x_t'
    try:
        result = parser.load(filepath)
        assert 'KEY' in result
    except FileNotFoundError:
        pytest.skip("Test file not found")


def test_extract_boundary_points():
    """Test extracting boundary points"""
    parser = ParasolidParser()
    try:
        parser.load('1pinflue4wire.x_t')
        points = parser.extract_boundary_points()
        assert points is not None
    except FileNotFoundError:
        pytest.skip("Test file not found")
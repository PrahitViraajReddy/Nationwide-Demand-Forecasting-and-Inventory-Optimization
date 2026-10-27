from pathlib import Path

import pytest

from demand_inventory import data


def _make_m5_layout(directory: Path, *, lfs_sales: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "calendar.csv").write_text("d,date\nd_1,2011-01-01\n", encoding="utf-8")
    (directory / "sell_prices.csv").write_text("store_id,item_id,wm_yr_wk,sell_price\nCA_1,item_1,11101,1.0\n", encoding="utf-8")
    sales = directory / "sales_train_validation.csv"
    if lfs_sales:
        sales.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:" + "a" * 64 + "\nsize 100\n",
            encoding="utf-8",
        )
    else:
        sales.write_text("id,d_1\nitem_1_CA_1_validation,1\n", encoding="utf-8")
    return directory


def test_resolves_m5_files_in_requested_directory(tmp_path):
    directory = _make_m5_layout(tmp_path / "m5")
    paths = data.validate_m5_files(directory)
    assert paths["sales"] == directory / "sales_train_validation.csv"
    assert paths["calendar"] == directory / "calendar.csv"
    assert paths["prices"] == directory / "sell_prices.csv"


def test_falls_back_to_repository_root_when_default_data_dir_is_missing(tmp_path, monkeypatch):
    _make_m5_layout(tmp_path)
    monkeypatch.setattr(data, "REPO_ROOT", tmp_path)
    assert data.resolve_data_dir(Path("data/raw/m5")) == tmp_path.resolve()


def test_lfs_pointer_produces_actionable_error(tmp_path):
    directory = _make_m5_layout(tmp_path / "m5", lfs_sales=True)
    with pytest.raises(RuntimeError, match="git lfs pull"):
        data.validate_m5_files(directory)

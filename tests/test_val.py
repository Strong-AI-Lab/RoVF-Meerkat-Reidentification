import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from training_functions.val import val


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)


def _anchor_fn(model, positive_list, negative_list, device, similarity_measure, is_ret_emb):
    del model, positive_list, negative_list, device, similarity_measure, is_ret_emb
    anchor = torch.tensor([0.0, 0.0])
    positive = torch.tensor([0.0, 0.1])
    selected_negative = torch.tensor([2.0, 2.0])
    all_negatives = [torch.tensor([2.0, 2.0]), torch.tensor([3.0, 3.0])]
    return anchor, positive, selected_negative, all_negatives


def _anchor_fn_top3_miss(model, positive_list, negative_list, device, similarity_measure, is_ret_emb):
    del model, positive_list, negative_list, device, similarity_measure, is_ret_emb
    anchor = torch.tensor([0.0, 0.0])
    positive = torch.tensor([10.0, 10.0])
    selected_negative = torch.tensor([0.1, 0.1])
    all_negatives = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    return anchor, positive, selected_negative, all_negatives


def _anchor_fn_one_negative(model, positive_list, negative_list, device, similarity_measure, is_ret_emb):
    del model, positive_list, negative_list, device, similarity_measure, is_ret_emb
    return (
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.1]),
        torch.tensor([2.0, 2.0]),
        None,
    )


def _anchor_fn_shape_mismatch(model, positive_list, negative_list, device, similarity_measure, is_ret_emb):
    del model, positive_list, negative_list, device, similarity_measure, is_ret_emb
    return (
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.1]),
        torch.tensor([[2.0, 2.0], [3.0, 3.0]]),
        None,
    )


def _similarity(query, gallery):
    return torch.linalg.norm(gallery - query, dim=1)


def test_val_without_metrics_returns_loss_only(tmp_path):
    criterion = nn.TripletMarginLoss(margin=1.0)
    valloader = [(["a", "b"], ["c"])]

    avg_loss, top_1, top_3 = val(
        DummyModel(),
        valloader,
        _anchor_fn,
        torch.device("cpu"),
        _similarity,
        criterion,
        str(tmp_path) + "/",
        batch_size=1,
        current_epoch=1,
        num_epochs=1,
        do_metrics=False,
    )

    assert avg_loss >= 0.0
    assert top_1 is None
    assert top_3 is None
    assert (tmp_path / "val_epoch_losses.txt").exists()


def test_val_with_metrics_returns_top1_and_top3(tmp_path):
    criterion = nn.TripletMarginLoss(margin=1.0)
    valloader = [(["a", "b"], ["c", "d"])]

    avg_loss, top_1, top_3 = val(
        DummyModel(),
        valloader,
        _anchor_fn,
        torch.device("cpu"),
        _similarity,
        criterion,
        str(tmp_path) + "/",
        batch_size=1,
        current_epoch=1,
        num_epochs=1,
        do_metrics=True,
    )

    assert avg_loss >= 0.0
    assert top_1 == 1.0
    assert top_3 == 1.0


def test_val_with_metrics_returns_zero_when_match_outside_top3(tmp_path):
    criterion = nn.TripletMarginLoss(margin=1.0)
    valloader = [(["a", "b"], ["c", "d", "e"])]

    avg_loss, top_1, top_3 = val(
        DummyModel(),
        valloader,
        _anchor_fn_top3_miss,
        torch.device("cpu"),
        _similarity,
        criterion,
        str(tmp_path) + "/",
        batch_size=1,
        current_epoch=1,
        num_epochs=1,
        do_metrics=True,
    )

    assert avg_loss >= 0.0
    assert top_1 == 0.0
    assert top_3 == 0.0


def test_val_empty_loader_fails(tmp_path):
    criterion = nn.TripletMarginLoss(margin=1.0)

    with pytest.raises(ValueError, match="no batches"):
        val(
            DummyModel(),
            [],
            _anchor_fn,
            torch.device("cpu"),
            _similarity,
            criterion,
            str(tmp_path) + "/",
            batch_size=1,
            current_epoch=1,
            num_epochs=1,
        )


def test_val_missing_negatives_fails(tmp_path):
    criterion = nn.TripletMarginLoss(margin=1.0)
    valloader = [(["a", "b"], [])]

    with pytest.raises(ValueError, match="Negative list"):
        val(
            DummyModel(),
            valloader,
            _anchor_fn,
            torch.device("cpu"),
            _similarity,
            criterion,
            str(tmp_path) + "/",
            batch_size=1,
            current_epoch=1,
            num_epochs=1,
        )


def test_val_metric_mode_requires_top3_gallery(tmp_path):
    criterion = nn.TripletMarginLoss(margin=1.0)
    valloader = [(["a", "b"], ["c"])]

    with pytest.raises(ValueError, match="at least 2 negative embeddings"):
        val(
            DummyModel(),
            valloader,
            _anchor_fn_one_negative,
            torch.device("cpu"),
            _similarity,
            criterion,
            str(tmp_path) + "/",
            batch_size=1,
            current_epoch=1,
            num_epochs=1,
            do_metrics=True,
        )


def test_val_shape_mismatch_fails(tmp_path):
    criterion = nn.TripletMarginLoss(margin=1.0)
    valloader = [(["a", "b"], ["c", "d"])]

    with pytest.raises(ValueError, match="matching shapes"):
        val(
            DummyModel(),
            valloader,
            _anchor_fn_shape_mismatch,
            torch.device("cpu"),
            _similarity,
            criterion,
            str(tmp_path) + "/",
            batch_size=1,
            current_epoch=1,
            num_epochs=1,
        )

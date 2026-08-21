# Phase 1 data protocol

## Canonical event schema

Every source dataset must map into:

| Field | Type | Meaning |
|---|---|---|
| `timestamp_ms` | integer | event time in Unix milliseconds |
| `source_row` | integer | deterministic source-order diagnostic only |
| `user_id` | string | anonymized user/visitor identifier |
| `item_id` | string | source item identifier |
| `behavior` | enum | `view`, `addtocart`, or `transaction` |
| `transaction_id` | optional string | source transaction identifier |

The loader streams rows instead of loading the full source CSV into memory.

## Split rule

We use two global timestamps:

```text
timestamp < train_end                         -> train
train_end <= timestamp < validation_end      -> validation
validation_end <= timestamp                  -> test
```

This is intentionally stricter than independently taking each user's last two
events. It prevents the training corpus from containing globally later events
than validation/test targets.

For a target at time `t`, its history contains all permitted events from the
same user with timestamps strictly less than `t`. Events sharing `t` cannot see
one another because the dataset does not establish a causal order within a
millisecond. Validation and test histories may include earlier train and
validation events; their labels are never inserted before prediction.

By default, targets whose items never appeared in the training period are
excluded from the in-vocabulary benchmark. Cold-start evaluation must be
reported separately and must opt in explicitly.

## RetailRocket adapter status

The optional adapter maps the published `events.csv` columns:

```text
timestamp, visitorid, event, itemid, transactionid
```

The source data card reports `view`, `addtocart`, and `transaction` events over
about 4.5 months. It also labels the dataset CC BY-NC-SA 4.0. Therefore:

- do not redistribute the raw files in this repository;
- do not automatically download them in default scripts;
- confirm the intended research/commercial use before adopting it as the formal
  enterprise benchmark;
- keep the core pipeline dataset-neutral so another approved dataset can be
  substituted without changing model code.

Source: https://www.kaggle.com/retailrocket/ecommerce-dataset/home


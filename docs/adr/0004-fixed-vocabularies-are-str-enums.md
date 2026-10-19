# Fixed vocabularies are StrEnums in tv.enums; strings are accepted at every boundary and written on every output

Every closed set of strings the package compares, defaults to or checks is a `StrEnum` in one module, `tactifoot_vision/enums.py` (`tv.enums`): roles, object kinds, the keys of `sv.Detections.data`, splits, the modes of single functions (annotator style, space-control model, overlay position, link mode, pooling, colour space, reducer), the built-in registry names and the run file sections. Package code uses the members instead of literals. The module imports nothing else from the package, so any module can import it.

A `StrEnum` member equals its string, so the boundaries stay strings:

- **In.** A parameter that takes a value from a set is typed `Enum | str` and turns the value into the member on entry with `Enum.parse(value, name)`. An unknown value raises a `ValueError` that names the parameter and lists the allowed values. Run files and the CLI pass plain strings and are unchanged (ADR 0001).
- **Out.** Tables, CSV, Parquet and JSON hold the plain values. A member pickles as its plain string, so a saved result loads without `tv.enums`, and it prints as its plain string, so messages and logs show what a user would type.
- **Open sets stay strings.** A detector's class name is any string (glossary: **Class name**); only the code that maps one to a **Role** uses `Role`, and a name outside the set is `Role.OTHER`. Output table columns, external formats (SoccerNet, Ultralytics, StatsBomb) and devices stay strings too.

## Considered options

- **`Literal[...]` types.** Rejected. They give the type checker the set but nothing at runtime: each function still needs its own hand-written check and message, and the set is restated wherever it is used. This repo configures no type checker, so the enum is what makes the set one definition that code, IDE and error messages share.
- **Closing the registries to their enums.** Rejected. The registries stay keyed by `str` so a plugin can register any name. The built-in names get an enum each (`ModelType`, `TrackerType`, `EmbedderType`, `PostProcessName`), and a test keeps each enum equal to its registry, so a new built-in backend without a member fails CI.

## Consequences

- Adding a role or a mode is one change in `tv.enums`; the call sites and the error messages follow.
- A member prints and pickles as its plain string. Debugging output shows `'player'`, not `<Role.PLAYER: 'player'>`; `type()` still tells them apart.

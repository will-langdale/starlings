# API reference

::: starlings
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        show_submodules: true
        filters:
            - "!^[A-Z]$"  # Excludes single-letter uppercase variables (like T, P, R)
            - "!^_"       # Excludes private attributes
            - "!^starlings$"  # Excludes the PyO3 module function
            - "!^generators$"  # Has its own dedicated page
            - "!^metrics$"  # Has its own dedicated page
            - "!^Metrics$"  # Metrics instance has its own page
            - "!^config$"  # Has its own dedicated page
            - "!^expressions$"  # Has its own dedicated page
            - "!^logger$"  # Private/undocumented
            - "!^logging$"  # Private/undocumented
            - "!^DEBUG_ENABLED$"  # Part of config module
            - "!^generate_entity_resolution_edges$"  # Part of generators module
            - "!^PyCollection$"  # Internal PyO3 class
            - "!^PyEntityFrame$"  # Internal PyO3 class
            - "!^PyPartition$"  # Internal PyO3 class
            - "!^Any$"  # Type annotation
            - "!^Iterable$"  # Type annotation
            - "!^annotations$"  # Module internals
            - "!^cast$"  # Type annotation
            - "!^tqdm$"  # External dependency
            - "!^version$"  # Module metadata
            - "!^debug$"  # Debug class

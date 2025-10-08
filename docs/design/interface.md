# Starlings Python interface

This document provides the complete API reference, integration guides, and usage patterns for the Starlings Python library. It shows how to use Starlings for entity resolution workflows, from basic threshold exploration to complex multi-stage hierarchical resolution.

## API design philosophy

The Starlings API follows the Polars pattern of separating data containers from operations through composable expressions. This enables intuitive, chainable operations whilst maintaining performance through lazy evaluation and pushed-down optimisations.

## Complete API reference

### Python API

The Starlings API design is inspired by [Polars](https://pola.rs/), adopting its expression-based approach for composable, efficient data operations. Like Polars, we provide a clear separation between data containers (EntityFrame/Collection) and operations (expressions), enabling users to build complex analyses from simple, chainable components. This design philosophy emphasises being opinionated about the "right way" to do things whilst maintaining flexibility for advanced use cases.

```python
import starlings as sl  # Standard import convention used throughout
```

### Core types

```python
class Key:
    """
    A flexible key type that can hold integers, strings, or bytes.
    
    Automatically converts based on input type:
    - int → stored as u32 or u64 based on size
    - str → stored as string
    - bytes → stored as bytes
    
    Example:
        Key(123)         # u32
        Key("cust_abc")  # string
        Key(b"\\x00\\x01")  # bytes
    """

class Entity:
    """
    A resolved entity containing records from one or more sources.
    
    Note: Entity objects are created by Starlings when you access partition.entities.
    You can also construct them for hierarchical resolution workflows.
    Entity is a Python-level type with no Rust counterpart.
    
    Properties:
        id: Optional[Key] - ID for referencing in edges (required for hierarchical resolution)
        members: Set[Tuple[str, Key]] - Set of (source, key) pairs
        size: int - Number of records in this entity
        sources: Set[str] - Unique sources contributing to this entity
    
    Example:
        # For viewing resolved entities
        entity.members = {
            ("CRM", Key(123)),
            ("CRM", Key(456)),
            ("MailingList", Key("user_abc"))
        }
        entity.size = 3
        entity.sources = {"CRM", "MailingList"}
        
        # For hierarchical resolution
        entity = Entity(
            id=Key(0),  # To reference in edges
            members={("src1", Key("key1")), ("src1", Key("key2"))}
        )
    """
```

### Core EntityFrame operations

```python
import starlings as sl

class EntityFrame:
    """
    Main container for multiple entity resolution collections over shared records.
    
    Mathematically: F = (R, {H₁, H₂, ..., Hₙ}, I) where collections ARE hierarchies.
    """
    
    @classmethod
    def from_records(cls, source_name: str, 
                    data: Union[pd.DataFrame, pa.Table, List[Dict]],
                    key_column: Optional[str] = None) -> 'EntityFrame':
        """
        Create EntityFrame from records.
        
        Args:
            source_name: Name of the source (e.g., "CRM", "MailingList")
            data: Records as DataFrame, Arrow Table, or list of dicts
            key_column: Column to use as key (default: auto-generate)
            
        Returns:
            New EntityFrame containing the records
            
        Example:
            ef = sl.from_records("CRM", df_customers, key_column="customer_id")
        """
    
    def add_collection_from_edges(self,
                                 name: str,
                                 edges: List[Tuple[int, int, float]]) -> None:
        """
        Add entity resolution collection from weighted edges.
        
        Uses the EntityFrame's complete record set to ensure isolated records
        are represented as singleton entities.
        
        Note: PyO3 automatically converts the Python list to &PyList for optimal
        batch processing with a single GIL acquisition in Rust.
        
        Args:
            name: Unique name for this collection
            edges: List of (record_i, record_j, similarity) tuples
            
        Example:
            ef.add_collection_from_edges("splink_v1", splink_edges)
        """
    
    def add_collection_from_entities(self,
                                    name: str,
                                    entities: Union[List[Entity], List[Set[Key]]]) -> None:
        """
        Add collection from pre-resolved entities at threshold 1.0.
        
        For probabilistic entity formation, use add_collection_from_edges instead.
        
        Args:
            name: Unique name for this collection
            entities: Either Entity objects or sets of keys representing resolved groups
            
        Example:
            # From sets of keys
            entities = [
                {"cust_123", "cust_456"},
                {"user_xyz", "user_abc"}
            ]
            ef.add_collection_from_entities("ground_truth", entities)
            
            # From Entity objects (e.g., from another collection)
            entities = other_partition.entities
            ef.add_collection_from_entities("imported", entities)
        """
    
    def add_collection(self, name: str, collection: 'Collection') -> None:
        """
        Add an existing Collection object to the frame.
        
        The collection must be standalone (not a view from another frame).
        If needed, use collection.copy() first.
        
        Args:
            name: Unique name for this collection
            collection: Existing Collection object
            
        Example:
            standalone = other_ef["results"].copy()
            ef.add_collection("imported", standalone)
        """
    
    def __getitem__(self, name: str) -> 'Collection':
        """
        Get collection by name for direct operations.
        
        Returns an immutable view of the collection that shares the frame's DataContext.
        Views cannot be modified directly - use EntityFrame methods for modifications.
        
        Example:
            partition = ef["splink"].at(0.85)
            sweep_df = ef["splink"].sweep(0.5, 0.95)
            
            # Views are immutable
            view = ef["splink"]
            # view.add_edges(...)  # Would raise an error
            
            # Create mutable copy if needed
            owned = ef["splink"].copy()
        """
    
    def analyse(self, *expressions, metrics: Optional[List] = None) -> List[Dict[str, Any]]:
        """
        Universal analysis method using expressions.

        Always returns List[Dict[str, Any]] in a DataFrame-friendly format where each dict
        represents one measurement point. This uniform schema works for both single-collection
        statistics and multi-collection comparison metrics.

        Args:
            *expressions: One or more sl.col() expressions. For asymmetric metrics (precision,
                         recall, f1), use .reference() to explicitly mark the ground truth,
                         or the last expression will be used as reference implicitly.
            metrics: List of metrics to compute. If None, defaults are:
                    - For comparisons (2+ collections): f1, precision, recall, ari, nmi
                    - For single collection: entity_count, entropy
                    The Rust layer always receives explicit metrics (Python provides defaults).

        Returns:
            List[Dict[str, float]]: DataFrame-friendly format where each dict contains:
            - `{collection}_threshold`: Threshold value for each collection (float)
            - Metric keys directly: `f1`, `precision`, `recall`, `ari`, `entity_count`, etc. (float)

            This format is optimised for immediate use with DataFrame libraries (polars, pandas)
            without needing to pivot or reshape the data.

        Example:
            # Point comparison (explicit reference)
            >>> result = ef.analyse(
            ...     sl.col("splink").at(0.85),
            ...     sl.col("truth").at(1.0).reference(),
            ...     metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision]
            ... )
            [{"splink_threshold": 0.85, "truth_threshold": 1.0,
              "f1": 0.92, "precision": 0.89}]

            # Single collection sweep
            >>> result = ef.analyse(
            ...     sl.col("splink").sweep(0.7, 0.9, 0.1),
            ...     metrics=[sl.Metrics.stats.entity_count]
            ... )
            [{"splink_threshold": 0.7, "entity_count": 1250},
             {"splink_threshold": 0.8, "entity_count": 980},
             {"splink_threshold": 0.9, "entity_count": 750}]

            # Sweep x point (implicit reference - last expression)
            >>> result = ef.analyse(
            ...     sl.col("splink").sweep(0.8, 0.9, 0.1),
            ...     sl.col("truth").at(1.0),  # Implicitly becomes reference
            ...     metrics=[sl.Metrics.eval.f1, sl.Metrics.stats.entity_count]
            ... )
            [{"splink_threshold": 0.8, "truth_threshold": 1.0,
              "f1": 0.91, "entity_count": 9870},
             {"splink_threshold": 0.9, "truth_threshold": 1.0,
              "f1": 0.94, "entity_count": 8540}]

            # Easy DataFrame conversion
            >>> import polars as pl
            >>> df = pl.from_dicts(result)
            >>> # Filter and analyse directly
            >>> df.filter(pl.col("f1") > 0.9)
        """
    
    # American spelling alias
    analyze = analyse
    
    def drop(self, *names: str) -> None:
        """
        Remove collections from the frame.
        
        Follows pandas/polars convention. Triggers automatic compaction
        when garbage exceeds 50% of total records.
        
        Args:
            *names: Names of collections to drop
            
        Example:
            ef.drop("splink_v1", "splink_v2")
        """
    
    def to_arrow(self) -> pa.Table:
        """
        Export EntityFrame to Arrow format.
        
        Uses dictionary encoding for efficient storage of repeated strings.
        
        Returns:
            Arrow Table with complete frame data
        """
    
    @classmethod
    def from_arrow(cls, table: pa.Table) -> 'EntityFrame':
        """
        Load EntityFrame from Arrow format.
        
        Args:
            table: Arrow Table created by to_arrow()
            
        Returns:
            Reconstructed EntityFrame
        """
```

### Collection operations

```python
class Collection:
    """
    Hierarchical partition structure that generates entities at any threshold.
    
    Collections exist in two states:
    - Standalone: Owns its DataContext exclusively (created via from_* methods)
    - View: Immutable view sharing DataContext with parent EntityFrame (created via ef["name"])
    
    Views cannot be modified. To create a mutable standalone collection from a view,
    use the explicit copy() method.
    """
    
    @property
    def is_view(self) -> bool:
        """Whether this collection is an immutable view from a frame."""
    
    @classmethod
    def from_edges(cls,
                  edges: List[Tuple[Any, Any, float]],
                  records: Optional[Union[List[Key], List[Entity]]] = None,
                  quantise: int = 6) -> 'Collection':
        """
        Build collection from weighted edges.
        
        Args:
            edges: List of (record_i, record_j, similarity) tuples
                  Records can be any hashable type (int, str, bytes)
                  At the Python/Rust boundary, these are converted to u32 indices
                  for efficient hierarchy construction
            records: Optional specification of complete record space:
                    - List[Key]: Records that should exist (can be all records or just 
                                additional ones not in edges - Rust deduplicates)
                    - List[Entity]: Pre-grouped entities for hierarchical resolution.
                                   All entity-to-edge expansion happens in Rust where
                                   all pairs within each entity get weight 1.0
                    - None: Only include records mentioned in edges
            quantise: Decimal places for threshold precision (1-6, default: 6).
                     Prevents floating-point comparison issues.
            
        Returns:
            New Collection with hierarchy of merge events
            
        Note:
            Complexity: O(m log m) where m = len(edges).
            Python keys (int/str/bytes) are converted to internal Key enum at the Python/Rust boundary.
            
        Example:
            # Basic usage
            edges = [
                ("cust_123", "cust_456", 0.95),
                (123, 456, 0.85),
                (b"hash1", b"hash2", 0.75)
            ]
            collection = sl.Collection.from_edges(edges)
            
            # With isolated records
            collection = sl.Collection.from_edges(
                edges=[(0, 1, 0.9), (2, 3, 0.8)],
                records=[0, 1, 2, 3, 4]  # Record 4 is an isolate
            )
            
            # Hierarchical resolution with pre-grouped entities
            # Entity expansion happens automatically in Rust
            entities = [
                Entity(id=Key(0), members={("src1", "key1"), ("src1", "key2")}),
                Entity(id=Key(1), members={("src2", "key3")}),
            ]
            collection = sl.Collection.from_edges(
                edges=[(0, 1, 0.9)],  # Links entity 0 to entity 1
                records=entities
            )
        """
    
    @classmethod
    def from_entities(cls,
                     entities: Union[List[Entity], List[Set[Key]]],
                     threshold: float = 1.0) -> 'Collection':
        """
        Build collection from pre-resolved entities.
        
        Creates a hierarchy where these entities exist at threshold 1.0.
        The entity list should represent a complete partition (every record 
        appears in exactly one entity). Isolated records should be included
        as singleton entities.
        
        Internally, entities are converted to edges in Rust (all pairs within  
        each entity get weight 1.0) before building the hierarchy.
        
        For probabilistic entity formation, use from_edges instead.
        
        Args:
            entities: Either Entity objects or sets of keys
            threshold: Always 1.0 (deterministic entities)
            
        Returns:
            New Collection
            
        Example:
            # From sets of keys
            entities = [
                {"cust_123", "cust_456"},
                {"user_xyz", "user_abc"}
            ]
            collection = sl.Collection.from_entities(entities)
            
            # From Entity objects
            collection = sl.Collection.from_entities(partition.entities)
        """
    
    def at(self, threshold: float) -> 'Partition':
        """
        Get partition at specific threshold.
        
        Note:
            First call at threshold: O(m) reconstruction
            Subsequent calls: O(1) from cache
        
        Returns:
            Partition object with entities at that threshold
            
        Example:
            partition = collection.at(0.85)
            print(f"Number of entities: {len(partition.entities)}")
        """
    
    def sweep(self, start: float = 0.0, 
             stop: float = 1.0, 
             step: float = 0.01) -> List[Dict[str, Any]]:
        """
        Analyse collection across threshold range.
        
        Note:
            Uses incremental O(k) updates between thresholds.
        
        Returns:
            List of dictionaries, one per threshold
            
        Example:
            analysis = collection.sweep(0.5, 0.95)
            # Returns [{"threshold": 0.5, "entity_count": 100}, ...]
        """
    
    def copy(self) -> 'Collection':
        """
        Create an owned copy of this collection.
        
        If collection is a view from an EntityFrame, creates a deep copy
        with its own DataContext. This enables safe mutation without
        affecting the parent frame.
        
        Returns:
            New Collection with independent data
            
        Example:
            view = ef["splink"]  # Immutable view
            owned = view.copy()  # Mutable standalone collection
        """
```

### Expression API

```python
# Expression builders for analyse() method
class col:
    """
    Create a collection expression for analysis.
    
    Inspired by polars.col() for consistent API design.
    """
    
    def __init__(self, name: str):
        """
        Reference a collection by name.
        
        Example:
            sl.col("splink")
        """
    
    def at(self, threshold: float) -> 'Expression':
        """
        Specify threshold for this collection.
        
        Example:
            sl.col("splink").at(0.85)
        """
    
    def sweep(self, start: float, stop: float, step: float = 0.01) -> 'Expression':
        """
        Specify threshold range for sweeping.

        Args:
            start: Starting threshold (inclusive)
            stop: Ending threshold (inclusive)
            step: Step size between thresholds (default: 0.01)

        Note:
            For performance at production scale (1M+ edges), step sizes are constrained:
            - Minimum step: 0.05
            - Steps are rounded to nearest 0.05 multiple
            - Example: step=0.01 becomes step=0.05
            - Rationale: Prevents excessive threshold points in large-scale sweeps

        Example:
            sl.col("splink").sweep(0.5, 0.95, 0.05)  # Recommended for large datasets
            sl.col("splink").sweep(0.5, 0.95, 0.1)   # Faster, coarser granularity
        """

    def reference(self) -> 'Expression':
        """
        Mark this collection as the reference (ground truth) for asymmetric metrics.

        When computing asymmetric comparison metrics like precision, recall, and f1,
        one collection must be designated as the reference (ground truth). Collections
        not marked with .reference() are treated as predictions.

        If no collection is explicitly marked as reference, the last expression
        passed to analyse() is implicitly used as the reference.

        Example:
            # Explicit reference
            ef.analyse(
                sl.col("splink").sweep(0.8, 0.9, 0.1),
                sl.col("truth").at(1.0).reference(),
                metrics=[sl.Metrics.eval.recall]
            )

            # Implicit reference (last expression)
            ef.analyse(
                sl.col("splink").sweep(0.8, 0.9, 0.1),
                sl.col("truth").at(1.0),  # Becomes reference implicitly
                metrics=[sl.Metrics.eval.recall]
            )
        """

# Pre-defined metrics as module constants (injectable functions)
# Accessed via sl.Metrics.eval.* and sl.Metrics.stats.*
class Metrics:
    class eval:
        f1 = F1Metric()
        precision = PrecisionMetric()
        recall = RecallMetric()
        ari = ARIMetric()
        nmi = NMIMetric()
        v_measure = VMeasureMetric()
        bcubed_precision = BCubedPrecisionMetric()
        bcubed_recall = BCubedRecallMetric()
    
    class stats:
        entropy = EntropyMetric()
        entity_count = EntityCountMetric()
```

### Partition operations

```python
class Partition:
    """
    A partition of records into entities at a specific threshold.
    """
    
    @property
    def entities(self) -> Set[Entity]:
        """Set of entities in this partition (unordered)."""
    
    @property
    def num_entities(self) -> int:
        """Number of entities in this partition."""
    
    def map(self, func: Union[Callable, 'Operation']) -> Dict[int, Any]:
        """
        Apply function to each entity.
        
        Args:
            func: Either a built-in operation or callable
                  Built-ins: sl.Ops.hash.sha256, sl.Ops.compute.size, etc.
                  
        Returns:
            Dictionary mapping entity ID to result
            
        Example:
            # Built-in hash function (runs in parallel Rust)
            hashes = partition.map(sl.Ops.hash.sha256)
            
            # Custom Python function
            custom = partition.map(lambda e: len(e.members) * 2)
        """
    
    def to_list(self) -> List[Set[Tuple[str, Key]]]:
        """
        Export entities as list of sets.
        
        Returns:
            List where each element is a set of (source, key) tuples
            
        Example:
            entities = partition.to_list()
            # [
            #   {("CRM", Key(123)), ("CRM", Key(456))},
            #   {("MailingList", Key(789))}
            # ]
        """
```

## Built-in operations

```python
# Operations for partition.map() (parallel execution in Rust)
# Accessed via sl.Ops.hash.* and sl.Ops.compute.*
class Ops:
    class hash:
        sha256 = SHA256Op()
        sha512 = SHA512Op()
        md5 = MD5Op()
        blake3 = Blake3Op()
    
    class compute:
        size = SizeOp()      # Number of records in entity
        density = DensityOp() # Internal connectivity
        fingerprint = FingerprintOp() # MinHash or similar
```

## Error handling

```python
# Exception hierarchy
class StarlingsError(Exception):
    """Base exception for all Starlings errors"""

class InvalidThresholdError(StarlingsError):
    """Threshold outside [0, 1] range"""

# ... other specific error types
```

## Data shape specifications

### Input formats for from_records()

```python
# DataFrame input
df = pd.DataFrame({
    'customer_id': [1, 2, 3],      # Will be used as key if specified
    'name': ['Alice', 'Bob', 'Charlie'],
    'email': ['alice@ex.com', 'bob@ex.com', 'charlie@ex.com']
})
ef = sl.from_records("CRM", df, key_column='customer_id')

# List of dicts input
records = [
    {'id': 1, 'name': 'Alice', 'email': 'alice@ex.com'},
    {'id': 2, 'name': 'Bob', 'email': 'bob@ex.com'}
]
ef = sl.from_records("MailingList", records, key_column='id')

# Arrow Table input
table = pa.Table.from_pandas(df)
ef = sl.from_records("OrderSystem", table, key_column='customer_id')
```

### Input formats for collections

```python
# From edges (record pairs with similarities)
edges = [
    (0, 1, 0.95),  # Record 0 and 1 with 0.95 similarity
    (1, 2, 0.85),  # Record 1 and 2 with 0.85 similarity
    (0, 3, 0.75)   # Record 0 and 3 with 0.75 similarity
]
collection = sl.Collection.from_edges(edges)

# From edges with isolated records
edges = [(0, 1, 0.9), (2, 3, 0.8)]
collection = sl.Collection.from_edges(
    edges=edges,
    records=[0, 1, 2, 3, 4]  # Record 4 is an isolate
)

# From fixed entities (sets of keys)
entities = [
    {0, 1, 4},  # First entity contains records 0, 1, 4
    {2, 3},     # Second entity contains records 2, 3
    {5, 6, 7, 8}  # Third entity
]
collection = sl.Collection.from_entities(entities)

# From Entity objects (automatic edge expansion in Rust)
entities = [
    Entity(id=Key("e1"), members={("CRM", "key1"), ("CRM", "key2")}),
    Entity(id=Key("e2"), members={("MailingList", "key3")})
]
collection = sl.Collection.from_entities(entities)
```

## Performance characteristics

Clear distinction between different operation types:

| Operation | First Time | Cached | Notes |
|-----------|------------|---------|-------|
| **Hierarchy build** | O(m log m) | - | One-time construction from edges |
| **Partition at threshold** | O(m) | O(1) | LRU cache holds 10 partitions |
| **Metric update** | O(k) | - | k = affected entities between thresholds |
| **Memory usage** | ~60-115MB per 1M edges | - | Varies with merge complexity |

Where:
- m = number of edges
- k = number of entities affected by threshold change
- Cache size is fixed at 10 partitions per hierarchy in v1

## Key conversion strategy

For optimal performance, Keys undergo conversion at different points:
- **Python level**: `Key` objects can hold int, str, or bytes
- **Python/Rust boundary**: Keys convert directly to u32 indices for hierarchy operations
- **Rust level**: Work with primitive u32 indices for speed
- **Storage**: Only the DataContext maintains the actual Key values

This strategy minimises overhead during computation-intensive hierarchy operations whilst preserving the flexibility of the Python API.

## Arrow schema

### Serialisation format

```python
# EntityFrame Arrow schema with optimal dictionary encoding
schema = pa.schema([
    # Parallel arrays for maximum deduplication efficiency
    ("record_sources", pa.dictionary(pa.int16(), pa.string())),  # Global source dictionary
    ("record_keys", pa.dictionary(pa.int32(), pa.dense_union([   # Dictionary of unions!
        pa.field("u32", pa.uint32()),
        pa.field("u64", pa.uint64()),
        pa.field("string", pa.string()),  # No nested dictionary needed
        pa.field("bytes", pa.binary())
    ]))),
    ("record_indices", pa.uint32()),  # Simple array of indices
    
    # Collections with merge events
    ("collections", pa.list_(pa.struct([
        ("name", pa.dictionary(pa.int8(), pa.string())),
        ("merge_events", pa.list_(pa.struct([
            ("threshold", pa.float64()),
            # RoaringBitmaps are expanded to nested lists for Arrow serialisation
            ("merging_groups", pa.list_(pa.list_(pa.uint32()))),  # Nested lists for Vec<RoaringBitmap>
        ])))
    ]))),
    
    # Metadata
    ("version", pa.string()),
    ("created_at", pa.timestamp('ms'))
])
```

### Database decomposition

The Arrow format can be decomposed into relational tables:

```sql
-- Records table
CREATE TABLE records (
    record_index INTEGER PRIMARY KEY,
    source VARCHAR(255),
    key VARCHAR(255),
    INDEX idx_source_key (source, key)
);

-- Collections table  
CREATE TABLE collections (
    collection_id INTEGER PRIMARY KEY,
    name VARCHAR(255) UNIQUE
);

-- Merge events table
CREATE TABLE merge_events (
    merge_id INTEGER PRIMARY KEY,
    collection_id INTEGER REFERENCES collections(collection_id),
    threshold DOUBLE
);

-- Merge groups table (each group within a merge event)
CREATE TABLE merge_groups (
    group_id INTEGER PRIMARY KEY,
    merge_id INTEGER REFERENCES merge_events(merge_id)
);

-- Records in each merge group
CREATE TABLE merge_group_records (
    group_id INTEGER REFERENCES merge_groups(group_id),
    record_index INTEGER REFERENCES records(record_index),
    PRIMARY KEY (group_id, record_index)
);
```

## Hierarchical resolution workflow

When working with pre-grouped entities (e.g., from previous resolution stages), Starlings supports hierarchical resolution:

```python
# Stage 1: Resolve within sources
crm_edges = dedupe_crm_records()
crm_collection = sl.Collection.from_edges(crm_edges)
crm_entities = crm_collection.at(0.9).entities

mail_edges = dedupe_mail_records()
mail_collection = sl.Collection.from_edges(mail_edges)
mail_entities = mail_collection.at(0.85).entities

# Stage 2: Resolve across sources using entities
# Entities need IDs for edge references
entities_with_ids = [
    Entity(id=Key(i), members=e.members) 
    for i, e in enumerate(crm_entities + mail_entities)
]

# Create edges between entities (not individual records)
cross_source_edges = link_entities(entities_with_ids)

# Build hierarchy over entities
# When records contains Entity objects, Rust automatically expands them to edges
final_collection = sl.Collection.from_edges(
    edges=cross_source_edges,
    records=entities_with_ids  # Entities expand to their member records
)
```

This workflow enables multi-stage resolution where early stages handle within-source deduplication and later stages handle cross-source linkage.

## Integration with Splink

```python
import splink
import starlings as sl
import polars as pl

# Run Splink
linker = splink.Linker(df, settings)
linker.predict()

# Get edges from Splink (not clusters)
edges = linker.get_scored_comparisons()
edges = [(e['id_l'], e['id_r'], e['match_probability']) for e in edges]

# Import into Starlings
ef = sl.from_records("source", df)
ef.add_collection_from_edges("splink_output", edges)

# Find optimal threshold using polars-inspired API
sweep_results = ef.analyse(
    sl.col("splink_output").sweep(0.5, 0.95, 0.01),
    sl.col("ground_truth").at(1.0).reference(),  # Explicit reference
    metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision, sl.Metrics.eval.recall]
)
# Returns DataFrame-friendly format:
# [{"splink_output_threshold": 0.5, "ground_truth_threshold": 1.0,
#   "f1": 0.72, "precision": 0.68, "recall": 0.76}, ...]

# Convert to polars for analysis
df_results = pl.from_dicts(sweep_results)
# Find optimal threshold by F1 score
optimal_row = df_results.filter(
    pl.col("f1") == pl.col("f1").max()
).row(0, named=True)
print(f"Optimal threshold: {optimal_row['splink_output_threshold']:.2f} (F1={optimal_row['f1']:.3f})")
```

## Integration with er-evaluation

```python
import er_evaluation as er_eval
import starlings as sl

# Load data into Starlings
ef = sl.from_records("dataset", records)

# Add collections
ef.add_collection_from_edges("predicted", predicted_edges)
ef.add_collection_from_entities("truth", true_entities)

# Use Starlings's efficient sweep
sweep_results = ef.analyse(
    sl.col("predicted").sweep(0.5, 0.95),
    sl.col("truth").at(1.0),  # Implicit reference (last expression)
    metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision, sl.Metrics.eval.recall]
)
# Returns DataFrame-friendly format:
# [{"predicted_threshold": 0.5, "truth_threshold": 1.0,
#   "f1": 0.82, "precision": 0.85, "recall": 0.79}, ...]

# Or extract partition for er-evaluation
partition = ef["predicted"].at(0.85)
clusters = partition.to_list()
metrics = er_eval.evaluate(clusters, true_clusters)

# Also supports American spelling
comparison = ef.analyze(
    sl.col("predicted").at(0.85),
    sl.col("truth").at(1.0).reference(),  # Explicit reference
    metrics=[sl.Metrics.eval.f1]
)
# Returns: [{"predicted_threshold": 0.85, "truth_threshold": 1.0, "f1": 0.89}]
```

## Integration with Matchbox

```python
import starlings as sl
import polars as pl
from matchbox import MatchboxClient

# Get Matchbox results
client = MatchboxClient()
matches = client.get_matches(dataset_id)

# Convert to edges for Starlings
edges = [(m['record_a'], m['record_b'], m['score']) for m in matches]

# Create EntityFrame
ef = sl.from_records("dataset", records)
ef.add_collection_from_edges("matchbox", edges)

# Find optimal threshold
sweep_results = ef.analyse(
    sl.col("matchbox").sweep(0.7, 0.95),
    sl.col("truth").at(1.0).reference(),
    metrics=[sl.Metrics.eval.f1]
)
# Returns tidy-row format for easy analysis
# Convert to polars and find optimal
df_results = pl.from_dicts(sweep_results)
optimal_threshold = df_results.filter(
    pl.col("metric_value") == pl.col("metric_value").max()
).select("collection_threshold")[0, 0]

# Export with hashes at optimal threshold
partition = ef["matchbox"].at(optimal_threshold)
hashes = partition.map(sl.Ops.hash.sha256)

matchbox_data = {
    'entities': partition.to_list(),
    'hashes': hashes,
    'threshold': optimal_threshold
}
client.upload_results(matchbox_data)
```

## Apache Arrow integration

```python
import pyarrow as pa
import pyarrow.parquet as pq
import starlings as sl

# Save to Arrow/Parquet
ef = sl.from_records("source", data)
# ... add collections ...

arrow_table = ef.to_arrow()
pq.write_table(arrow_table, "starlings_frame.parquet")

# Load from Arrow/Parquet
table = pq.read_table("starlings_frame.parquet")
ef = sl.EntityFrame.from_arrow(table)

# The Arrow format uses optimal dictionary encoding for efficiency
# Sources and keys are automatically deduplicated at the global level
```

## Implementation roadmap

### Phase 1: Foundation (Months 1-3)
**Goal**: Core functionality with single collection

- Week 1-2: Rust core data structures
  - Merge event storage implementation
  - Basic hierarchy with reconstruction
  - DataContext with append-only guarantee
  
- Week 3-4: Python bindings
  - PyO3 setup with starlings package
  - Basic from_records and Collection.from_edges
  - Collection.at() for partition access
  
- Week 5-8: Essential algorithms
  - Connected components with merge event extraction
  - Basic metrics (precision, recall, F1)
  - Partition reconstruction from merges
  - Proper handling of isolated records
  
- Week 9-12: Testing and optimisation
  - Comprehensive test suite
  - Performance benchmarks
  - Memory profiling

**Deliverable**: Working single-collection Starlings with basic Python API

### Phase 2: Multi-collection & analytics (Months 4-6)
**Goal**: Multiple collections and comprehensive metrics

- Week 1-3: Multi-collection support
  - EntityFrame with multiple collections
  - DataContext sharing via Arc
  - ef.analyse() with expressions
  - Immutable view pattern for collections
  
- Week 4-6: Complete metrics suite
  - ARI, NMI, V-measure implementation
  - B-cubed metrics (semi-incremental)
  - Injectable metric functions
  - Required metrics in Rust API
  
- Week 7-9: Incremental computation
  - Metric state tracking between thresholds
  - O(k) updates for adjacent thresholds
  - Computation statistics
  
- Week 10-12: Analysis tools
  - Threshold sweep optimisation
  - Collection.sweep() implementation
  - Entity lifetime tracking

**Deliverable**: Full multi-collection comparison capabilities

### Phase 3: Batch performance optimisation (Months 7-9)
**Goal**: Production-ready batch processing performance

- Week 1-3: Parallelisation
  - Rayon integration
  - Parallel merge event extraction
  - Concurrent collection comparison
  - partition.map() with parallel built-ins
  
- Week 4-6: SIMD optimisations
  - Vectorised metric computation
  - SIMD-accelerated set operations
  - Cache-optimised layouts
  - Built-in hash operations (SHA256, BLAKE3)
  
- Week 7-9: Memory optimisation
  - Automatic compaction on ef.drop()
  - Collection.copy() for explicit detachment
  - Memory-mapped storage for large datasets
  - Mandatory quantisation support
  
- Week 10-12: Batch processing enhancements
  - Optimised cache strategies
  - Batch comparison operations
  - Performance profiling and tuning

**Deliverable**: 10-100x performance improvement for batch operations

### Phase 4: Integration & polish (Months 10-12)
**Goal**: Ecosystem integration and production features

- Week 1-3: Tool integrations
  - Splink adapter
  - er-evaluation compatibility
  - Matchbox export/import with hashing
  
- Week 4-6: Advanced features
  - Custom metric plugins
  - Collection.from_entities() with type unions
  - Advanced Arrow serialisation
  - Hierarchical resolution support
  
- Week 7-9: Production tooling
  - Monitoring and metrics
  - Error handling and recovery
  - Configuration management
  
- Week 10-12: Documentation & release
  - Complete API documentation  
  - Tutorial notebooks
  - Performance guides
  - v1.0 release preparation

**Deliverable**: Production-ready Starlings 1.0

## Performance benchmarks

### Target performance metrics

```yaml
# Single machine performance targets
# Note: Assumes sparse graphs from blocking (m ~ n to n log n)

small_scale:  # 10K records, ~50K edges
  hierarchy_build: < 100ms
  cached_query: < 1ms
  uncached_query: < 10ms
  full_sweep_1000_thresholds: < 1s
  memory_usage: < 10MB

medium_scale:  # 1M records, ~5M edges
  hierarchy_build: < 10s
  cached_query: < 1ms
  uncached_query: < 200ms
  full_sweep_1000_thresholds: < 30s
  memory_usage: < 100MB

large_scale:  # 10M records, ~50M edges
  hierarchy_build: < 5min
  cached_query: < 10ms
  uncached_query: < 2s
  full_sweep_1000_thresholds: < 5min
  memory_usage: < 1GB

# Scaling characteristics
complexity:
  hierarchy_construction: O(m log m) where m = edges
  partition_reconstruction: O(m)
  cached_query: O(1)  # LRU cache maintains 10 partitions per hierarchy (non-configurable in v1)
  incremental_metric: O(k) where k = affected entities
  memory: O(m) for merge events + O(c × n) for c cached partitions
```

### Benchmark suite

```python
# Performance testing harness
import starlings as sl
import time
import memory_profiler

class StarlingsBenchmark:
    def __init__(self, num_records, num_edges):
        self.num_records = num_records
        self.num_edges = num_edges
        
    def benchmark_hierarchy_build(self):
        edges = self.generate_edges()
        start = time.time()
        collection = sl.Collection.from_edges(edges)  # Uses default quantise=6
        return time.time() - start
    
    def benchmark_partition_reconstruction(self, collection):
        # Clear cache to force reconstruction
        collection.clear_cache()
        start = time.time()
        partition = collection.at(0.85)
        return time.time() - start
    
    def benchmark_cached_access(self, collection):
        # Warm cache
        _ = collection.at(0.85)
        # Measure cached access
        start = time.time()
        partition = collection.at(0.85)
        return time.time() - start
    
    def benchmark_threshold_sweep(self, ef):
        start = time.time()
        results = ef.analyse(
            sl.col("test").sweep(0, 1, 0.001),
            sl.col("truth").at(1.0),
            metrics=[sl.Metrics.eval.f1]
        )
        return time.time() - start
    
    def benchmark_entity_hashing(self, collection):
        partition = collection.at(0.85)
        start = time.time()
        hashes = partition.map(sl.Ops.hash.blake3)
        return time.time() - start
```

## Migration guide

### From pandas DataFrame resolution

```python
# Before: Manual threshold management
def resolve_entities(df, threshold):
    clusters = []
    for t in [0.7, 0.8, 0.9]:
        clusters.append(cluster_at_threshold(df, t))
    return clusters[threshold]

# After: Starlings with full hierarchy
import starlings as sl

ef = sl.from_records("data", df)
edges = compute_edges(df)  # Your existing edge computation
ef.add_collection_from_edges("resolved", edges)

# Now can query any threshold instantly (after first reconstruction)
partition = ef["resolved"].at(0.7)  # O(m) first time, O(1) after
partition = ef["resolved"].at(0.85)  # O(1) if cached

# Efficient sweep with incremental updates
sweep_results = ef.analyse(
    sl.col("resolved").sweep(0.5, 0.95),
    sl.col("truth").at(1.0).reference(),
    metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision]
)  # Returns tidy-row List[Dict], O(k) between thresholds

# Can convert to polars if needed
import polars as pl
df = pl.from_dicts(sweep_results)
```

### From multiple resolution attempts

```python
# Before: Separate scripts for each method
splink_results = run_splink(data, params1)
dedupe_results = run_dedupe(data, params2)
comparison = compare_results(splink_results, dedupe_results, truth)

# After: Unified Starlings
import starlings as sl

ef = sl.from_records("data", data)

# Add as edges, not clusters
ef.add_collection_from_edges("splink", splink_edges)
ef.add_collection_from_edges("dedupe", dedupe_edges)
ef.add_collection_from_entities("truth", truth_entities)

# Compare systematically using expressions
comparison = ef.analyse(
    sl.col("splink").at(0.85),
    sl.col("dedupe").at(0.76),
    sl.col("truth").at(1.0).reference(),  # Explicit reference
    metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision, sl.Metrics.eval.recall]
)
# Returns DataFrame-friendly format:
# [{"splink_threshold": 0.85, "dedupe_threshold": 0.76, "truth_threshold": 1.0,
#   "f1": 0.89, "precision": 0.92, "recall": 0.86},
#  {"splink_threshold": 0.85, "dedupe_threshold": 0.76, "truth_threshold": 1.0,
#   "f1": 0.85, "precision": 0.88, "recall": 0.82}, ...]

# Find optimal thresholds efficiently
splink_sweep = ef.analyse(
    sl.col("splink").sweep(0.5, 0.95),
    sl.col("truth").at(1.0),  # Implicit reference (last expression)
    metrics=[sl.Metrics.eval.f1]
)
# Convert to dataframe for analysis
import polars as pl
df = pl.from_dicts(splink_sweep)
optimal = df.filter(pl.col("f1") == pl.col("f1").max()).row(0, named=True)
```

### Important notes on performance

1. **First partition access**: Reconstructs from merge events O(m)
2. **Subsequent accesses**: Use cache O(1)
3. **Threshold sweeps**: Use incremental updates O(k) between adjacent thresholds
4. **Memory usage**: ~60-115MB for 1M edges with DataContext architecture
5. **Sparse graphs assumed**: Starlings expects m << n² from blocking/LSH
6. **Key types**: All key types (int/str/bytes) have equivalent performance after interning
7. **Collection views**: Views from EntityFrames are immutable for safety
8. **Isolated records**: Handled automatically in EntityFrame context

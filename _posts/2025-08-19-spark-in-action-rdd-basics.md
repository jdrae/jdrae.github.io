---
title: "Spark in Action 1: From MapReduce to RDDs"
date: 2025-08-19
categories: [data-engineering]
tags: [spark, hadoop, mapreduce, rdd]
translation_key: spark-in-action-rdd-basics
---

I am going to record my notes from reading Spark in Action by Petar Zečević and Marko Bonaći in three parts. In this first post, I will start with MapReduce and Hadoop as background for understanding Spark, then summarize Spark's basic execution flow and the concept of RDDs.

---

## What Is MapReduce?

MapReduce is a large-scale data processing model introduced in Google's paper *MapReduce: Simplified Data Processing on Large Clusters*. Its core idea is to make cluster computing easier to handle through a simpler model.

The MapReduce processing flow can be viewed in three broad steps.

1. Split a job into smaller pieces and map them across multiple nodes in a cluster for distributed processing.
2. Each node processes the task assigned to it and produces intermediate results.
3. The split intermediate results are aggregated in the reduce phase to produce the final result.

MapReduce tries to solve three major problems.

- **Parallel processing**: split work into smaller units and process them simultaneously.
- **Data distribution**: split data across multiple nodes for storage and processing.
- **Fault tolerance**: handle failures in distributed components.

For example, the master periodically sends pings to all worker nodes. If a worker does not respond for a certain period of time, the master determines that the worker has a problem, resets the map tasks that worker was handling to their initial state, and reschedules them on another worker.

An important idea in this model is not to move data to where computation happens, but to **send the program to where the data is stored**. For large-scale data, network transfer costs are high, so it is important to compute as close to the data as possible.

### Word Count Example

The most representative example is word count.

1. **map**: split each sentence into words and return a list of `(word, 1)` pairs.
2. **shuffle phase**: group map results by key so that the same word is passed to the same reducer.
3. **reduce**: sum the occurrences for each word to produce the final result.

The shuffle phase can become a bottleneck, but it makes aggregation by word simple in the subsequent reduce phase.

## What Is Spark?

Spark is a big data processing platform that replaces Hadoop's MapReduce.

Hadoop is a Java-based open-source framework for distributed computing. People usually think of it together with the Hadoop Distributed File System, or HDFS, and the MapReduce processing engine.

Spark is similar to Hadoop in that it is a general-purpose distributed computing platform. However, because it is designed to keep large amounts of data in memory, better performance can be expected for iterative computation or interactive analysis.

In Hadoop MapReduce, if the result of one job needs to be used in another job, it must be saved to HDFS and then read again. This makes it inefficient for iterative algorithms. Also, not every problem can be naturally decomposed using only MapReduce operations.

Spark can be viewed as a processing engine that emerged to address these limitations.

### Cases Where Spark Is Not Suitable

Spark is not the right tool for every situation.

Because it uses a distributed architecture, some overhead occurs in processing time. This overhead is not a major problem for large datasets, but for small datasets, another framework may be more efficient.

Spark is also not suitable for OLTP systems, which process large volumes of atomic transactions. Instead, it is better suited for batch processing or analytical workloads, namely OLAP.

## Hadoop's Core Ideas

Hadoop is based on three main ideas.

- **Parallelization**: split many operations into smaller parts.
- **Distribution**: split data across multiple nodes for storage.
- **Fault tolerance**: handle failures in distributed components.

Spark shares these basic assumptions of distributed processing. The difference lies in how data is reused and how execution plans are constructed.

## Spark's Execution Process

Suppose we store a 300 MB file in an HDFS cluster. HDFS can split this file into blocks of 128 MB, 128 MB, and 44 MB, and store them across three nodes in the cluster. If the replication factor is set to the default value of 3, HDFS also replicates each block to two other nodes.

Spark asks Hadoop for the location of each block, or partition, of the file. It then loads each block into the RAM of the HDFS node where that block is stored. This is called **data locality**.

Using data locality allows computation to happen near where the data exists, rather than moving large amounts of data over the network.

The distributed collection referenced by an RDD is a set of multiple partitions. Users do not need to think every time about the fact that this collection is split across multiple nodes.

For example, when filtering is performed, only the filtered information is stored in RAM. If `cache` is used afterward, the same RDD can be reused in memory by another job without loading the file again. This filtering operation runs in parallel across multiple nodes.

## RDD

RDD stands for Resilient Distributed Dataset. It is Spark's basic abstraction and the core concept for handling data in a distributed environment.

RDDs have three major characteristics.

### Immutability

An RDD is a read-only dataset. Transformation operators do not modify an existing RDD directly; they always create a new RDD object. In other words, once an RDD is created, it is immutable.

### Resilience

An RDD has fault tolerance. Even if a node fails, the RDD can be restored.

Spark records the log of transformation operators used to create a dataset. If a failure occurs, it does not rebuild the entire dataset. Instead, it recomputes only the dataset held by the failed node and restores the RDD.

### Distribution

An RDD is a dataset stored on one or more nodes. Users can use it like a logical collection without directly handling which physical node stores the data.

This can be understood as **location transparency**. Even if the physical pieces of a file are stored in multiple places, users access the data through a file name or RDD reference.

## Transformation Operators and Action Operators

Spark operations can be broadly divided into transformation operators and action operators.

- **Transformation operators**: manipulate data and create a new RDD. Examples include `filter` and `map`.
- **Action operators**: actually return computation results. Examples include `count` and `foreach`.

Spark uses **lazy evaluation**. Calling a transformation operator does not immediately trigger computation. Actual computation is executed when an action operator is called.

Thanks to this approach, Spark can collect execution plans and compute them in a more efficient way.

## Scala for Comprehension Example

The book also covers Scala code. For example, the following code reads lines from a file and creates a `Set`.

```scala
val employees = Set() ++ (
  for {
    line <- fromFile(empPath).getLines
  } yield line.trim
)
```

At each cycle of the `for` loop, the `line.trim` value is added to a temporary collection. When the loop ends, this temporary collection is returned and then merged into the `Set`.

## Shared Variables

In a distributed environment, multiple nodes in a cluster sometimes need to refer to the same data. In this case, Spark's shared variables can be used.

```scala
val bcEmployees = sc.broadcast(employees)
val isEmp = user => bcEmployees.value.contains(user)
```

Shared variables are sent exactly once to each node in the cluster and automatically cached in memory. If shared variables are not used, the same data may be repeatedly transferred over the network as many times as the number of tasks performing the work.

Spark distributes shared variables using a P2P protocol. Each node exchanges and spreads the shared variable with other nodes, which is also called a gossip protocol. This prevents the master execution from being significantly delayed.

When accessing a shared variable, the `value` method must be used.

## Closing

In this post, I first looked at MapReduce and Hadoop as background for understanding Spark, then summarized Spark's basic execution flow and RDDs.

In the next post, I will summarize partitioning, shuffling, and RDD dependencies, which are important for understanding Spark performance.

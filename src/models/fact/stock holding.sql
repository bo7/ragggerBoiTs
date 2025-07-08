{{ config(materialized='view') }}
select * from {{ source('fact', 'Stock Holding') }}

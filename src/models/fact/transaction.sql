{{ config(materialized='view') }}
select * from {{ source('fact', 'Transaction') }}

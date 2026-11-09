-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) as total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: Three-table join (Revenue by city and product category)
SELECT c.city, p.category, SUM(o.order_amount) as revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category;

-- Q3: Scalar subquery (Products priced above average catalogue price)
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN (Customers with no completed orders)
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: Common Table Expression (CTE) for customer spend above average
WITH customer_totals AS (
    SELECT customer_id, SUM(order_amount) as total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) as mean_spend FROM customer_totals
)
SELECT ct.customer_id, ct.total_spend
FROM customer_totals ct, avg_spend as av
WHERE ct.total_spend > av.mean_spend;

-- Q6: Ranking window functions partitioned by city
SELECT c.city, o.customer_id, SUM(o.order_amount) as total_spend,
       RANK() OVER (PARTITION BY c.city ORDER BY SUM(o.order_amount) DESC) as rnk,
       DENSE_RANK() OVER (PARTITION BY c.city ORDER BY SUM(o.order_amount) DESC) as dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY c.city ORDER BY SUM(o.order_amount) DESC) as row_num
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.city, o.customer_id;

-- Q7: Running total of daily completed revenue
WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) as day, SUM(order_amount) as rev
    FROM orders
    WHERE status = 'completed'
    GROUP BY day
)
SELECT day, rev,
       SUM(rev) OVER (ORDER BY day) as cumulative_revenue
FROM daily_revenue;

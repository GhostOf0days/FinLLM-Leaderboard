========================================================================
XBRL Analytics
========================================================================

.. contents:: Table of Contents
   :local:

Failure Case: ROA Calculation
----------------------------------------

Analysis of Coca-Cola's Return on Assets (ROA) in FY2023 using XBRL filings.

File Sources
~~~~~~~~~~~~

- 10-K filings: https://www.sec.gov/edgar/browse/?CIK=21344&owner=exclude
- XBRL instance file: https://www.sec.gov/Archives/edgar/data/21344/000002134424000009/ko-20231231_htm.xml

.. figure:: /use_cases/images/failure_XBRL_analytics_1_full_file_1.png
   :width: 100%
   :alt: XBRL parsing failure

   **Parsing Failure**:

   - Incorrect contextRef format used (expected "c-n" pattern)
   - Failed to extract values from XML structure

.. figure:: /use_cases/images/failure_XBRL_analytics_1_full_file_2.png
   :width: 100%
   :alt: Context identification failure

   **Context Errors**:

   - Wrong contextRef "c-31" instead of correct "c-1"
   - Noise-induced confusion in financial contexts

.. figure:: /use_cases/images/failure_XBRL_analytics_1_section.png
   :width: 100%
   :alt: Asset recognition error

   **Asset Value Errors**:

   - FY2023: contextRef "c-23" ($97,703M) vs model output
   - FY2022: contextRef "c-26" ($92,763M) vs model output

Validation Testing
~~~~~~~~~~~~~~~~~~
1. **Closed-book Testing**:

   - Model failed to provide ROA without data
   - Correct formula recall: Net Income / Average Assets

2. **Ticker Recognition**:

   - Successfully mapped "KO" to Coca-Cola
   - Still lacked financial data access

3. **Online Search**:

   - Retrieved 10.97% from Stock Data Online
   - Formula discrepancy: website vs standard calculation

4. **Formula-Guided Search**:

   - Correct calculation: 11.25% using Macrotrends data
   - Validated formula: (NI_2023)/((Assets_2022+Assets_2023)/2)

Failure Case: Revenue Forecasting
---------------------------------
3-year revenue prediction using 2019-2023 growth rates.

.. figure:: /use_cases/images/failure_XBRL_analytics_3.png
   :width: 100%
   :alt: Revenue forecast errors

   **Calculation Errors**:

   - Growth rate rounding: -11.41% → -11.4%
   - Average growth miscalculation: 5.84% → 5.34%
   - Compound errors in 2025-2026 projections

Financial Statements Reference
------------------------------
.. figure:: /use_cases/images/failure_XBRL_statements_all.png
   :width: 100%
   :alt: Coca-Cola financial statements

   Source financial data for all analyses:

   a) Income Statement 2023
   b) Balance Sheet 2023
   c) Income Statement 2022
   d) Income Statement 2021


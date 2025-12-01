Skip to content
Chat history




You said:
let's say for UBS work, I want to build an evals framework. The agent is a bit multi stage so we need to think of the right way to do this. The idea is that there are 13 forms types (UBS client internal forms) and each of them has a layout model + ocr extractor pipeline (or a custom azure doc extractor that combines the two stages) and then runs through some use cases (i.e. validate that this matches data in the DB, or it matches some format or whatever). We need to build an evals around that. This is my thinking. 1) we create a directory of all of our test files (the PDFS that they pull data from) and test DB calls i guess as well (these can be mocked or jsons of real client data pulled from DBs, or in the tests we can do run time DB calls). It doesn't matter. But we can say for each form, for each use case, if it gets it right its a 1, otherwise its a 0. And build a score. that can handle the total evaluations. sep we need to build the train test validate sets for each document right? for azure custom extractor what is the right # of samples to start train with? 
ChatGPT said:

Thought for 46s
You’re on the right track: treat each (form_type, use_case) as a discrete eval task, score it 0/1 at the top level, and build everything around that. The only thing I’d change is: make the plumbing a bit more structured and be more deliberate about how many labeled docs you feed into Azure DI.

I’ll break it into:

How to structure the eval harness for a multi-stage pipeline

How to organize data / directories

Metrics and scoring (beyond 0/1)

How many samples for Azure custom extractor + how to split train/val/test

1. Shape of the eval framework
Think in three layers of evaluation:

Layer A – Unit evals: extraction quality
This is “how good is the layout + OCR + custom extractor before any DB checks or business logic?”

For each form type (one of your 13):

Define a canonical schema, e.g. for W-9:

tax_id, name, address, signature_present, date, TIN_type, DSID, etc.

For a subset of PDFs, create gold labels:

Either label in Document Intelligence Studio or via your own label tool and store a JSON like:

{
  "form_type": "W9",
  "fields": {
    "tax_id": "123-45-6789",
    "name": "Jane Doe",
    "signature_present": true,
    "dsid": "ABC123"
  }
}
Your extraction service (layout + custom model + normalization) returns the same schema:

{
  "form_type": "W9",
  "fields": {
    "tax_id": {"value": "123-45-6789", "confidence": 0.98},
    "name": {"value": "Jane Doe", "confidence": 0.99},
    ...
  }
}
Eval: Compare predicted vs gold on a per-field basis (accuracy, F1, coverage). This is where you see how much DI is helping per form before DB logic gets in the way.

Layer B – Integration evals: pipeline logic
Now you take “DI output + internal logic,” but still treat it as a service:

Input:

PDF path

DB fixture (mocked record) or a way to look up real DB

Use case definition (e.g. validate_tax_id_matches_db)

Output:

A structured result:

{
  "use_case": "validate_tax_id_matches_db",
  "status": "pass" | "fail",
  "reason": "TAX_ID_MISMATCH",
  "details": {...}
}
Eval: For each test case, you have an expected status (and possibly reason).
Your top-level metric: pass → 1, fail → 0 on this outcome.

So your mental model is:

Layer A: “Did we read the form correctly?”

Layer B: “Given what we read + DB state, did the business rule fire correctly?”

Both are important and can regress independently.

Layer C – End-to-end agent evals
If you have a multi-step agent (e.g. picks the form type, calls the right DI model, then runs validation rules, then writes back status), you treat the entire agent as a black box:

Input: PDF + client_id

Expected outcome: some state in your system (e.g. form_validated = true, or “case opened for missing signature”).

Eval: again, binary pass/fail per use case.

This lets you:

Use Layer A to tune DI and detect extraction regressions.

Use Layer B to test pipeline logic.

Use Layer C as a final gate for agent behavior.

2. Data & directory layout
You’re basically right: a filesystem-backed test corpus is the simplest.

Something like:

/forms_eval/
  /W9/
    /unit_extraction/
      pdfs/
        w9_001.pdf
        w9_002.pdf
      labels.jsonl        # gold field values per PDF
    /use_cases/
      /validate_tax_id_matches_db/
        cases.jsonl       # list of {pdf_path, db_fixture_id, expected_status}
        db_fixtures.jsonl # mocked DB rows
      /check_signature_present/
        ...
  /SIG_PAGE/
    ...
Key design points:

Canonical IDs: each PDF has a stable ID (w9_001), referenced from labels and test cases.

DB fixtures:

Either:

JSON fixture rows stored alongside tests, and the pipeline under test is wired to read from a fixture “DB” in test mode

Or:

A dedicated test database that you seed before running the suite; tests reference primary keys.

Form type registry:

One place (YAML/JSON/Pydantic) where you enumerate the 13 forms, their fields, and applicable use cases.

Your eval harness just reads this registry and discovers tests.

This gives you a stable spine: when you change DI models, forms, or rules, your tests stay the same.

3. Metrics & scoring
Your “1 if correct else 0” idea is exactly what you want for the top-line MD/manager score, but internally you’ll want more signal.

For unit extraction
Per form type:

Field-level accuracy / F1:

exact_match(field) tracked across the dataset.

Coverage:

Percentage of cases where a value was produced at all (no null/empty).

Per-field confusion modes:

Missing vs wrong vs mislocalized vs format error (e.g. TIN format wrong but digits correct).

You can still roll this up into a simple number like:

extraction_score = (# of field matches across all cases) / (total # of fields)
But you’ll want the breakdown when something regresses.

For per-use-case pipeline evals
Per (form_type, use_case):

use_case_accuracy = passed_cases / total_cases (your 0/1 metric)

Optionally:

Precision/recall if the use case is detection-like (e.g. “detect missing signature” where you have positive and negative examples).

Then:

Overall score:

overall_score = Σ (w_form,use_case * use_case_accuracy)
with weights if some forms/use cases are business-critical (e.g. W-9 TIN vs some low-risk format check).

Error taxonomy
You’ll thank yourself later if you tag failures with a small, fixed error taxonomy:

EXTRACTION_MISS (DI didn’t read it correctly)

NORMALIZATION_ERROR (DI read it but your post-processing mangled it)

DB_LOOKUP_ERROR (DB fixture / mapping failed)

RULE_LOGIC_ERROR (rule implemented incorrectly)

AGENT_ROUTING_ERROR (wrong form type or wrong pipeline selected)

Those tags let your dashboard slice failures by where they originate.

4. Azure custom extractor: how many samples?
What Microsoft says (hard minimums)
For custom extraction / custom document models, Microsoft’s official guidance:

You can start training with as few as 5 documents of the same form type.
Microsoft Learn
+1

For composed models, they recommend five or more labeled files per document type for reliable training.
Microsoft Learn

Community guidance and Q&A threads suggest aiming for 5–10 labeled samples per template to get better generalization.
Microsoft Learn

Some tutorials use 5 training + 1 test as the bare-bones setup.
Twyzer | Smart data-driven solutions

So: the platform will happily train with 5; that’s the platform minimum, not the UBS production target.

What I’d actually do for UBS-grade forms
Given your setting (13 fixed, highly structured internal forms; lots of historical PDFs; high accuracy requirements), I’d structure it like this:

Phase 0 – Bring-up / feasibility
Goal: get something working quickly per form type.

Per form type:

~10 labeled docs total:

8 train

2 internal sanity-check (not your real test set; just to see if it does anything sane)

Use custom neural models for anything with noticeable variation; template models only if the layout is literally identical every time.

This lets you:

Validate that DI can handle each form at all.

Integrate the model into your pipeline and eval harness.

Phase 1 – First real model + real eval
Per form type:

~30 labeled docs:

20 train

5 validation (for you to inspect performance and tweak)

5 reserved for the unit extraction test set (never used in training)

You still keep a separate pipeline-level test set (see below). This is the point where:

You run your full unit extraction eval on those 5 held-out docs.

You start to see stable per-field metrics.

Phase 2 – Production-level stability
If you can afford it, I’d grow toward 50–100 labeled docs per form type over time:

~70–80 train

10 validation

10–20 test (unit extraction test, per form)

Why that range?

Your forms will have:

Scanner vs Docusign vs in-branch capture

Slight layout versions over the years

Handwriting / stamps / marginalia

You want the custom model to see that diversity; 5–10 documents won’t expose it.

The good news: custom neural models are designed to adapt quickly; you don’t need thousands per form. A few dozen, well-diversified samples per form is usually enough for strong performance.

How to split train / val / test holistically
For each form type:

Collection: Gather as many historical PDFs as you reasonably can (hundreds if possible).

Stratify by:

Layout version

Capture type (scan vs Docusign)

Language, if relevant

Choose 3 sets:

Training set for DI:

The subset of labeled docs you let DI learn from.

Dev/validation set:

Labeled docs you do not train DI on, used to tune thresholds, normalization logic, and to compare alternative DI versions.

Test set(s):

Unit extraction test: same schema, used to measure DI extraction quality.

Pipeline use-case test: may be a superset, where you also need DB fixtures and expected use-case outcomes (pass/fail).

You can reuse some PDFs across unit extraction and pipeline test sets if you must, but the cleaner pattern:

Reserve a subset of docs exclusively for final evaluation, especially if you’ll be reporting these numbers to MD-level stakeholders.

Quick answer to your last line
“sep we need to build the train test validate sets for each document right? for azure custom extractor what is the right # of samples to start train with?”

Yes: you want explicit train / val / test splits per form type, not “per document” but per form template. For 13 forms, that’s 13 little mini-datasets with their own splits.

For Azure custom extractor:

Hard minimum to get going: 5–10 labeled docs per form.

Practical sweet spot to aim for in Phase 1: ~20–30 labeled docs per form.

Longer-term production target: 50–100 labeled docs per form (or more if you see a lot of variation).

If you want, next step we can sketch a concrete TestCase / UseCaseSpec dataclass layout and a small YAML schema for the form registry so you can plug this straight into your existing Python test harness.

You said:
you can think of the pipeline as preprocessing + extraction + validation. I mean in the case of azure DI i suppose that preprocessing is very minimal, its just extraction + validation. But yes I agree. Can you create a document that describes these 3 layers? if we use azure doc intelligence (for labeling in the DI studio) and then as the custom extractor, and then how we go about setting up the actual both training data and then the evals data for this. I like your 3 layers 
ChatGPT said:
To tailor the document properly, could you confirm:

What level of technical detail should it include? (e.g. suitable for engineering teams vs stakeholders)

Should it include YAML schema or code examples for the eval framework?

Are there internal compliance or data handling constraints I should note in describing training/eval setup with Azure DI?

Should the document be oriented around UBS internal use only, or should it remain tool-agnostic enough to be reused elsewhere?

You said:
1. A mix. Imagine I am writing it. So my level of technical abstraction, but that I am explaining it to both devs and product people. 2. Sure you can but leave it towards the end after a broad data 3. nto really. 4. No UBS use 
ChatGPT said:
Understood. I’ll draft a document that outlines the three-layer evaluation framework (Preprocessing → Extraction → Validation) using Azure Document Intelligence, detailing how UBS can structure both training and evaluation datasets. It will include schema examples at the end. I’ll let you know when the document is ready for review—you can continue working in the meantime.

Research completed in 7m · 14 sources · 25 searches
ChatGPT said:
Document Processing Pipeline: Preprocessing, Extraction, and Validation
Overview
This document describes a three-layer pipeline for intelligent document processing, comprised of Preprocessing, Extraction, and Validation stages. The pipeline is designed to convert raw documents into reliable structured data using Azure AI Document Intelligence (formerly Azure Form Recognizer) as the core extraction engine. We outline each layer’s role and how they work together, and discuss how to prepare model training and evaluation datasets. The goal is to present a clear technical approach that’s understandable to both developers and product stakeholders.
Preprocessing
In the preprocessing stage, we prepare incoming documents for analysis. In our case, Azure Document Intelligence supports a wide range of file formats (PDFs, images like JPEG/PNG, Office documents, etc.) natively
learn.microsoft.com
, so minimal pre-processing is required. Key preprocessing tasks include:
Format and Quality Checks: Ensure the document is in a supported format and is legible. Azure’s service can handle scans and photos, but providing high-quality, clear documents improves accuracy
sparkco.ai
. For example, if a PDF is password-protected, remove the password before processing
learn.microsoft.com
. Verify that images/PDFs are correctly oriented (no upside-down pages) and not excessively noisy or blurry.
Document Ingestion: Load or convert the document into the pipeline. This could involve simple steps like reading a PDF file or image from storage. Azure’s Document Intelligence will accept the raw file without needing manual image preprocessing (the service’s OCR and layout analysis will handle it)
stackoverflow.com
. In other words, we can send the document “as is” and rely on Azure to perform any needed OCR, rotation, or layout detection internally.
For most cases, preprocessing is lightweight. We focus on feeding the extraction layer with the best possible input (clear scans, correct file types), but we don’t need complex transformations since the Azure AI handles that. Using high-quality inputs can boost extraction accuracy significantly
sparkco.ai
, so our preprocessing simply ensures documents meet basic quality and format criteria.
Extraction
The extraction layer is the heart of the pipeline. Here we use Azure AI Document Intelligence to automatically extract structured data from the documents. We train a custom extraction model tailored to our specific document types and fields of interest, since our use-case involves forms or documents not fully covered by Azure’s prebuilt models. The extraction process involves two parts: building the custom model, and using it at runtime:
Model Training: To create a custom extractor, we first gather a set of sample documents and label them with the fields we want to extract. Using the Azure Document Intelligence Studio, we define field names and manually highlight the corresponding text in each document (e.g. Invoice Number, Date, Total Amount, etc.). Azure requires a handful of labeled documents (at least 5 to get started) to train a custom model
learn.microsoft.com
. In practice, we aim for a larger and more diverse training set (e.g. dozens of documents) for better accuracy
learn.microsoft.com
sparkco.ai
. The Studio saves these labels (generating JSON label files for each document), and then we initiate training of a custom model. We typically choose a custom neural model (for unstructured or variable layouts) unless the documents have a fixed form layout where a template model would suffice. After labeling, training the model is as simple as clicking “Train” in the Studio, which triggers Azure to learn the field patterns from our annotated data
medium.com
. The result is a trained model with a unique Model ID that we can call via API.
Runtime Extraction: Once the custom model is deployed, the extraction layer takes each new incoming document and calls Azure’s Document Intelligence API to analyze it with our model. Azure’s service will perform OCR and layout understanding on the document, then apply our trained model to identify and extract the defined fields. The output is a structured result (often in JSON) containing field names, the extracted values, and confidence scores for each
medium.com
. For example, given a new form, the service might return "InvoiceNumber": "12345" with 98% confidence, "TotalAmount": "$1,000" with 95% confidence, etc. This is essentially the machine reading stage – the unstructured document is transformed into structured data. We encapsulate the results into our internal data structure (sometimes referred to as a Dossier of extracted fields and values). If there are tables or line items to extract, the model can capture those as well (maintaining row/column structure), though our primary focus is on key fields.
Using Azure’s cloud AI for extraction means we don’t have to build OCR or AI models from scratch – we leverage Microsoft’s AI. It can handle multi-page documents, various file types, and even handwritten text if our model is trained for it. The custom model is tuned to our specific documents, which improves accuracy over generic OCR. In summary, the Extraction layer automates data capture, producing a set of structured fields from each document, along with metadata like confidence scores and bounding box coordinates (if needed for provenance).
Validation
After extraction, the validation layer checks and refines the extracted data to ensure it meets our quality and business requirements. This stage acts as a safeguard – it verifies that the data output by the AI makes sense and flags any issues before the data is used downstream. The Validation layer performs several types of checks:
Business Rule Validation: We apply rules to validate each extracted field against expected patterns or ranges. For example, check that dates are in a valid format, amounts are positive numbers, and required fields (like an account number or name) are not missing. If a field should match a certain regex or fall within a known list of values, we verify that. This rules-based step catches obvious errors (e.g., a phone number field containing letters) and ensures compliance with formats. It essentially validates extracted data against expected formats and business logic
onyxdata.co.uk
.
Confidence Thresholds: We leverage the confidence scores from the extraction model to decide if data is trustworthy. If a key field was extracted with very low confidence (for instance, 40% confidence on an ID number), the validation layer can flag this for manual review or mark the whole document’s output as needing attention. We set a threshold (say 90%) for automatic pass; anything below that might be routed for additional verification. This way, the pipeline can automatically route low-confidence cases to a human or an exception queue based on confidence thresholds
onyxdata.co.uk
, ensuring critical errors don’t slip through.
Cross-Field Consistency: The validation may also compare related fields for consistency. For instance, if the document has a total amount field and a sum of line-item amounts, we can cross-check that they match. Similarly, if one field is a date of birth and another is an age, we could verify they are consistent. Any discrepancies indicate a potential extraction error that needs review.
Aggregation of Validation Results: The outcome of this stage is typically a validation report or status. Each document’s data is annotated with validation results – e.g., all checks passed, or which fields failed which checks. We may compute an overall validation score or simply mark the document as “valid” or “needs review”. These results can be stored for auditing and to help improve the extraction model over time (for example, if certain fields frequently fail validation, we might add more training examples or adjust the model).
The Validation layer ensures that the structured data is reliable and ready for use. In cases where everything passes, the data can automatically flow into downstream systems (for example, uploaded into a database or sent to an onboarding workflow). If there are validation issues, the system can either automatically correct minor issues or escalate for human review. This regular validation of outputs is crucial to maintain accuracy and trust
sparkco.ai
 – it serves as a feedback loop to catch errors. In fact, implementing validation checks can significantly reduce downstream data entry errors (studies suggest by ~25% in some scenarios)
sparkco.ai
. Thus, validation is an indispensable part of the pipeline, especially for critical business documents, as it guarantees that the automated extraction is meeting the required quality standards.
Model Training and Evaluation Data Setup
Building an effective extraction pipeline requires careful preparation of training and evaluation datasets for the custom AI model. We outline how we set up the data for model training and how we evaluate the model’s performance:
Collect Representative Training Documents: We gather a diverse set of documents that cover the various layouts and content variations we expect in production. For example, if processing invoices from multiple vendors, we include samples from each vendor. We aim for at least 5 documents to start (Azure’s minimum)
learn.microsoft.com
, but typically use dozens. It’s important that all target fields appear and are labeled in these samples (if a field sometimes is blank in a document, we include other samples where it’s filled)
learn.microsoft.com
. Using a diverse set of samples improves model learning
sparkco.ai
, so our training set includes different scenarios (e.g. different form versions, varied values in fields, high and low quality scans, etc.). We also ensure documents are high quality or as clear as possible, because training on clearer text helps the model learn better.
Label the Training Data in Azure DI Studio: Using the Azure Document Intelligence Studio, we create a new Custom Project for our model. We upload the training documents (often to an Azure Blob Storage container which the studio connects to). In the labeling interface, we define each field name we want to extract, then go through each document and highlight the text corresponding to each field. The studio makes this easy with an OCR view of the document; we just click on the text for, say, “Company Name” or “Invoice Total” and assign it to the field. We repeat this for all fields across all training documents until each document’s fields are labeled. The studio generates a .labels.json (and accompanying OCR JSON) for each file to record our annotations
learn.microsoft.com
. Once labeling is complete, we have a fully labeled dataset. This label data, along with the original documents, forms the training dataset that Azure will use.
Train the Custom Model: After labeling, we trigger the training process in Document Intelligence Studio. We select “Train”, provide a model name/ID, and choose the model type (custom neural or custom template). The training runs in Azure’s cloud – no code needed from our side – and typically completes within an hour or less for a modest dataset
medium.com
. Azure’s machine learning will find patterns that map the visual text to our defined fields. Once training finishes successfully, our custom model is available in our Azure resource with the given Model ID. We record this Model ID for use in the extraction stage of our pipeline (so our code knows which custom model to call). The trained model’s quality is heavily dependent on the quality and representativeness of the training data, so the previous steps are crucial. Azure Document Intelligence requires surprisingly few samples to get started, but more data generally yields better accuracy, especially for complex documents. Best practices suggest using ~5 samples at minimum and up to 50+ if possible, especially if documents have variability
learn.microsoft.com
. We also note any training-time metrics or warnings from Azure; for instance, it might report if certain fields had insufficient examples.
Prepare an Evaluation Dataset: Equally important is setting aside some evaluation (test) documents. These are documents not used in training, but which we have ground-truth labels for (we might label a few extra docs and not include them in the training set). This simulates new, unseen documents. We use this evaluation set to assess how well the model generalizes. For example, if we collected 50 samples, we might train on 40 and reserve 10 as evaluation, or use k-fold cross validation if data is very limited. In practice, we’ll often label everything and let Azure train on all, then do manual testing on additional docs; but when possible, having a dedicated test set is ideal for an unbiased evaluation. Each eval document has known true values for the fields (since we labeled them offline or know the answers), so we can compare the model’s output to the ground truth.
Evaluate Model Performance: We run the newly trained model on the evaluation documents (either through the Studio’s “Analyze” function or via the API in our pipeline test environment)
medium.com
. We then measure accuracy by comparing extracted fields to the expected values. We look at metrics like field-level accuracy (percentage of fields correctly extracted), and possibly precision/recall if partial matches are an issue. For instance, if out of 100 total fields in the eval set the model filled 90 correctly, that’s 90% accuracy. We pay special attention to critical fields – e.g., the model might do well on easier fields like dates but struggle on a handwritten signature field. The evaluation helps identify such gaps. Azure’s JSON output includes confidence scores, which we also examine: are errors happening mostly when confidence is low (which we can catch via validation)? If the model is confident but wrong, that shows a need to improve training data. We document the performance for each field (e.g., “Field X has 95% accuracy, Field Y has 80% – possibly needs more training examples”). This stage is essentially our QA for the model.
Iterate and Improve: Based on the evaluation results, we may choose to refine the model. If certain field extractions are below acceptable accuracy, we can add more training samples focusing on those cases or adjust our labeling (ensure consistency, etc.). We then retrain and evaluate again. This iterative loop continues until the model meets our accuracy targets. It’s also a continuous process post-deployment: as new documents are processed, we can collect any that the model struggled with (especially those flagged in validation) and add them to the training set for the next model update. Azure’s platform allows us to update the model by re-training with an expanded dataset. Maintaining and updating the model with fresh data over time helps keep accuracy high as document styles change
sparkco.ai
.
By following this training and evaluation approach, we ensure that our custom extractor is both well-trained and validated before it’s put into production use. We have high confidence in what the model will extract correctly, and we have the Validation layer (described above) to catch the occasional errors. Together, a solid training/eval regime and the three-layer pipeline (Preprocessing → Extraction → Validation) result in a robust document processing solution. It transforms raw documents into structured, verified data ready for use in our business workflows, reducing manual effort and error rates while providing transparency into the process (through validation reports and evaluation metrics).






Sources



No file chosenNo file chosen
ChatGPT can make mistakes. Check important info.

Document Processing Pipeline: Preprocessing, Extraction, and Validation
Overview
This document describes a three-layer pipeline for intelligent document processing, comprised of Preprocessing, Extraction, and Validation stages. The pipeline is designed to convert raw documents into reliable structured data using Azure AI Document Intelligence (formerly Azure Form Recognizer) as the core extraction engine. We outline each layer’s role and how they work together, and discuss how to prepare model training and evaluation datasets. The goal is to present a clear technical approach that’s understandable to both developers and product stakeholders.
Preprocessing
In the preprocessing stage, we prepare incoming documents for analysis. In our case, Azure Document Intelligence supports a wide range of file formats (PDFs, images like JPEG/PNG, Office documents, etc.) natively
learn.microsoft.com
, so minimal pre-processing is required. Key preprocessing tasks include:
Format and Quality Checks: Ensure the document is in a supported format and is legible. Azure’s service can handle scans and photos, but providing high-quality, clear documents improves accuracy
sparkco.ai
. For example, if a PDF is password-protected, remove the password before processing
learn.microsoft.com
. Verify that images/PDFs are correctly oriented (no upside-down pages) and not excessively noisy or blurry.
Document Ingestion: Load or convert the document into the pipeline. This could involve simple steps like reading a PDF file or image from storage. Azure’s Document Intelligence will accept the raw file without needing manual image preprocessing (the service’s OCR and layout analysis will handle it)
stackoverflow.com
. In other words, we can send the document “as is” and rely on Azure to perform any needed OCR, rotation, or layout detection internally.
For most cases, preprocessing is lightweight. We focus on feeding the extraction layer with the best possible input (clear scans, correct file types), but we don’t need complex transformations since the Azure AI handles that. Using high-quality inputs can boost extraction accuracy significantly
sparkco.ai
, so our preprocessing simply ensures documents meet basic quality and format criteria.
Extraction
The extraction layer is the heart of the pipeline. Here we use Azure AI Document Intelligence to automatically extract structured data from the documents. We train a custom extraction model tailored to our specific document types and fields of interest, since our use-case involves forms or documents not fully covered by Azure’s prebuilt models. The extraction process involves two parts: building the custom model, and using it at runtime:
Model Training: To create a custom extractor, we first gather a set of sample documents and label them with the fields we want to extract. Using the Azure Document Intelligence Studio, we define field names and manually highlight the corresponding text in each document (e.g. Invoice Number, Date, Total Amount, etc.). Azure requires a handful of labeled documents (at least 5 to get started) to train a custom model
learn.microsoft.com
. In practice, we aim for a larger and more diverse training set (e.g. dozens of documents) for better accuracy
learn.microsoft.com
sparkco.ai
. The Studio saves these labels (generating JSON label files for each document), and then we initiate training of a custom model. We typically choose a custom neural model (for unstructured or variable layouts) unless the documents have a fixed form layout where a template model would suffice. After labeling, training the model is as simple as clicking “Train” in the Studio, which triggers Azure to learn the field patterns from our annotated data
medium.com
. The result is a trained model with a unique Model ID that we can call via API.
Runtime Extraction: Once the custom model is deployed, the extraction layer takes each new incoming document and calls Azure’s Document Intelligence API to analyze it with our model. Azure’s service will perform OCR and layout understanding on the document, then apply our trained model to identify and extract the defined fields. The output is a structured result (often in JSON) containing field names, the extracted values, and confidence scores for each
medium.com
. For example, given a new form, the service might return "InvoiceNumber": "12345" with 98% confidence, "TotalAmount": "$1,000" with 95% confidence, etc. This is essentially the machine reading stage – the unstructured document is transformed into structured data. We encapsulate the results into our internal data structure (sometimes referred to as a Dossier of extracted fields and values). If there are tables or line items to extract, the model can capture those as well (maintaining row/column structure), though our primary focus is on key fields.
Using Azure’s cloud AI for extraction means we don’t have to build OCR or AI models from scratch – we leverage Microsoft’s AI. It can handle multi-page documents, various file types, and even handwritten text if our model is trained for it. The custom model is tuned to our specific documents, which improves accuracy over generic OCR. In summary, the Extraction layer automates data capture, producing a set of structured fields from each document, along with metadata like confidence scores and bounding box coordinates (if needed for provenance).
Validation
After extraction, the validation layer checks and refines the extracted data to ensure it meets our quality and business requirements. This stage acts as a safeguard – it verifies that the data output by the AI makes sense and flags any issues before the data is used downstream. The Validation layer performs several types of checks:
Business Rule Validation: We apply rules to validate each extracted field against expected patterns or ranges. For example, check that dates are in a valid format, amounts are positive numbers, and required fields (like an account number or name) are not missing. If a field should match a certain regex or fall within a known list of values, we verify that. This rules-based step catches obvious errors (e.g., a phone number field containing letters) and ensures compliance with formats. It essentially validates extracted data against expected formats and business logic
onyxdata.co.uk
.
Confidence Thresholds: We leverage the confidence scores from the extraction model to decide if data is trustworthy. If a key field was extracted with very low confidence (for instance, 40% confidence on an ID number), the validation layer can flag this for manual review or mark the whole document’s output as needing attention. We set a threshold (say 90%) for automatic pass; anything below that might be routed for additional verification. This way, the pipeline can automatically route low-confidence cases to a human or an exception queue based on confidence thresholds
onyxdata.co.uk
, ensuring critical errors don’t slip through.
Cross-Field Consistency: The validation may also compare related fields for consistency. For instance, if the document has a total amount field and a sum of line-item amounts, we can cross-check that they match. Similarly, if one field is a date of birth and another is an age, we could verify they are consistent. Any discrepancies indicate a potential extraction error that needs review.
Aggregation of Validation Results: The outcome of this stage is typically a validation report or status. Each document’s data is annotated with validation results – e.g., all checks passed, or which fields failed which checks. We may compute an overall validation score or simply mark the document as “valid” or “needs review”. These results can be stored for auditing and to help improve the extraction model over time (for example, if certain fields frequently fail validation, we might add more training examples or adjust the model).
The Validation layer ensures that the structured data is reliable and ready for use. In cases where everything passes, the data can automatically flow into downstream systems (for example, uploaded into a database or sent to an onboarding workflow). If there are validation issues, the system can either automatically correct minor issues or escalate for human review. This regular validation of outputs is crucial to maintain accuracy and trust
sparkco.ai
 – it serves as a feedback loop to catch errors. In fact, implementing validation checks can significantly reduce downstream data entry errors (studies suggest by ~25% in some scenarios)
sparkco.ai
. Thus, validation is an indispensable part of the pipeline, especially for critical business documents, as it guarantees that the automated extraction is meeting the required quality standards.
Model Training and Evaluation Data Setup
Building an effective extraction pipeline requires careful preparation of training and evaluation datasets for the custom AI model. We outline how we set up the data for model training and how we evaluate the model’s performance:
Collect Representative Training Documents: We gather a diverse set of documents that cover the various layouts and content variations we expect in production. For example, if processing invoices from multiple vendors, we include samples from each vendor. We aim for at least 5 documents to start (Azure’s minimum)
learn.microsoft.com
, but typically use dozens. It’s important that all target fields appear and are labeled in these samples (if a field sometimes is blank in a document, we include other samples where it’s filled)
learn.microsoft.com
. Using a diverse set of samples improves model learning
sparkco.ai
, so our training set includes different scenarios (e.g. different form versions, varied values in fields, high and low quality scans, etc.). We also ensure documents are high quality or as clear as possible, because training on clearer text helps the model learn better.
Label the Training Data in Azure DI Studio: Using the Azure Document Intelligence Studio, we create a new Custom Project for our model. We upload the training documents (often to an Azure Blob Storage container which the studio connects to). In the labeling interface, we define each field name we want to extract, then go through each document and highlight the text corresponding to each field. The studio makes this easy with an OCR view of the document; we just click on the text for, say, “Company Name” or “Invoice Total” and assign it to the field. We repeat this for all fields across all training documents until each document’s fields are labeled. The studio generates a .labels.json (and accompanying OCR JSON) for each file to record our annotations
learn.microsoft.com
. Once labeling is complete, we have a fully labeled dataset. This label data, along with the original documents, forms the training dataset that Azure will use.
Train the Custom Model: After labeling, we trigger the training process in Document Intelligence Studio. We select “Train”, provide a model name/ID, and choose the model type (custom neural or custom template). The training runs in Azure’s cloud – no code needed from our side – and typically completes within an hour or less for a modest dataset
medium.com
. Azure’s machine learning will find patterns that map the visual text to our defined fields. Once training finishes successfully, our custom model is available in our Azure resource with the given Model ID. We record this Model ID for use in the extraction stage of our pipeline (so our code knows which custom model to call). The trained model’s quality is heavily dependent on the quality and representativeness of the training data, so the previous steps are crucial. Azure Document Intelligence requires surprisingly few samples to get started, but more data generally yields better accuracy, especially for complex documents. Best practices suggest using ~5 samples at minimum and up to 50+ if possible, especially if documents have variability
learn.microsoft.com
. We also note any training-time metrics or warnings from Azure; for instance, it might report if certain fields had insufficient examples.
Prepare an Evaluation Dataset: Equally important is setting aside some evaluation (test) documents. These are documents not used in training, but which we have ground-truth labels for (we might label a few extra docs and not include them in the training set). This simulates new, unseen documents. We use this evaluation set to assess how well the model generalizes. For example, if we collected 50 samples, we might train on 40 and reserve 10 as evaluation, or use k-fold cross validation if data is very limited. In practice, we’ll often label everything and let Azure train on all, then do manual testing on additional docs; but when possible, having a dedicated test set is ideal for an unbiased evaluation. Each eval document has known true values for the fields (since we labeled them offline or know the answers), so we can compare the model’s output to the ground truth.
Evaluate Model Performance: We run the newly trained model on the evaluation documents (either through the Studio’s “Analyze” function or via the API in our pipeline test environment)
medium.com
. We then measure accuracy by comparing extracted fields to the expected values. We look at metrics like field-level accuracy (percentage of fields correctly extracted), and possibly precision/recall if partial matches are an issue. For instance, if out of 100 total fields in the eval set the model filled 90 correctly, that’s 90% accuracy. We pay special attention to critical fields – e.g., the model might do well on easier fields like dates but struggle on a handwritten signature field. The evaluation helps identify such gaps. Azure’s JSON output includes confidence scores, which we also examine: are errors happening mostly when confidence is low (which we can catch via validation)? If the model is confident but wrong, that shows a need to improve training data. We document the performance for each field (e.g., “Field X has 95% accuracy, Field Y has 80% – possibly needs more training examples”). This stage is essentially our QA for the model.
Iterate and Improve: Based on the evaluation results, we may choose to refine the model. If certain field extractions are below acceptable accuracy, we can add more training samples focusing on those cases or adjust our labeling (ensure consistency, etc.). We then retrain and evaluate again. This iterative loop continues until the model meets our accuracy targets. It’s also a continuous process post-deployment: as new documents are processed, we can collect any that the model struggled with (especially those flagged in validation) and add them to the training set for the next model update. Azure’s platform allows us to update the model by re-training with an expanded dataset. Maintaining and updating the model with fresh data over time helps keep accuracy high as document styles change
sparkco.ai
.
By following this training and evaluation approach, we ensure that our custom extractor is both well-trained and validated before it’s put into production use. We have high confidence in what the model will extract correctly, and we have the Validation layer (described above) to catch the occasional errors. Together, a solid training/eval regime and the three-layer pipeline (Preprocessing → Extraction → Validation) result in a robust document processing solution. It transforms raw documents into structured, verified data ready for use in our business workflows, reducing manual effort and error rates while providing transparency into the process (through validation reports and evaluation metrics).
Citations

Build and train a custom model - Document Intelligence - Foundry Tools | Microsoft Learn

https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/how-to-guides/build-a-custom-model?view=doc-intel-4.0.0

Mastering Azure Form Recognition for Document Intelligence

https://sparkco.ai/blog/mastering-azure-form-recognition-for-document-intelligence

Build and train a custom model - Document Intelligence - Foundry Tools | Microsoft Learn

https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/how-to-guides/build-a-custom-model?view=doc-intel-4.0.0

ocr - Azure Formrecognizer - how to improve input image quality by performing preprocessing? - Stack Overflow

https://stackoverflow.com/questions/66004490/azure-formrecognizer-how-to-improve-input-image-quality-by-performing-preproce/76884245

Mastering Azure Form Recognition for Document Intelligence

https://sparkco.ai/blog/mastering-azure-form-recognition-for-document-intelligence

Build and train a custom model - Document Intelligence - Foundry Tools | Microsoft Learn

https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/how-to-guides/build-a-custom-model?view=doc-intel-4.0.0

Build and train a custom model - Document Intelligence - Foundry Tools | Microsoft Learn

https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/how-to-guides/build-a-custom-model?view=doc-intel-4.0.0

Mastering Azure Form Recognition for Document Intelligence

https://sparkco.ai/blog/mastering-azure-form-recognition-for-document-intelligence

Microsoft Azure AI Document Intelligence with Custom build Models: A Beginners Guide | by Dr Booma | Medium

https://medium.com/@dr.booma19/microsoft-azure-ai-document-intelligence-with-custom-build-models-a-beginners-guide-37ca6b16c7a9

Microsoft Azure AI Document Intelligence with Custom build Models: A Beginners Guide | by Dr Booma | Medium

https://medium.com/@dr.booma19/microsoft-azure-ai-document-intelligence-with-custom-build-models-a-beginners-guide-37ca6b16c7a9

Azure Document Intelligence: Breaking the Document Processing Barrier

https://onyxdata.co.uk/azure-document-intelligence-breaking-the-document-processing-barrier/

Mastering Azure Form Recognition for Document Intelligence

https://sparkco.ai/blog/mastering-azure-form-recognition-for-document-intelligence

Mastering Azure Form Recognition for Document Intelligence

https://sparkco.ai/blog/mastering-azure-form-recognition-for-document-intelligence

Build and train a custom model - Document Intelligence - Foundry Tools | Microsoft Learn

https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/how-to-guides/build-a-custom-model?view=doc-intel-4.0.0

Mastering Azure Form Recognition for Document Intelligence

https://sparkco.ai/blog/mastering-azure-form-recognition-for-document-intelligence
All Sources

learn.microsoft

sparkco

stackoverflow

medium

onyxdata.co

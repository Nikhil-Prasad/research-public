# UBS Internal MRM Validation Template — DeepSeek OCR on Genesis

## Document Control
- **Document owner:** [Name]
- **Business owner:** [Name]
- **Technical owner:** [Name]
- **Review date:** [Date]
- **Environment:** Genesis (UBS STAAT Azure subscription)
- **Model:** DeepSeek OCR [exact model/version]
- **Status:** Draft / Under Review / Approved

## 1. Purpose
This document records the internal MRM validation summary for UBS use of DeepSeek OCR hosted within the Genesis Azure environment.

The objective is to document:
- model provenance,
- internal hosting design,
- data boundary and network controls,
- operational setup, and
- key residual validation items required for sign-off.

## 2. Scope
This validation covers the internal deployment and use of DeepSeek OCR in Genesis, including:
- model artifact acquisition,
- internal storage of model files,
- Azure compute provisioning,
- Kubernetes-based model serving through vLLM, and
- controls preventing CID from leaving the Genesis environment.

Out of scope:
- external SaaS inference,
- internet-connected runtime environments,
- transfer of CID outside Genesis,
- any production design that requires outbound internet from serving infrastructure.

## 3. Implementation Summary
The current implementation was completed as follows:

1. DeepSeek OCR model artifacts were downloaded from `huggingface.co`.
2. The downloaded artifacts were extracted and saved to the internal Genesis file share.
3. Required Azure compute was provisioned inside Genesis, including VMs, GPUs, and related infrastructure.
4. The model was deployed and served internally using vLLM within Kubernetes pods.
5. All storage, compute, and inference remain inside Genesis. Runtime boxes do not have outbound internet access, and CID is not transferred outside Genesis.

**Control note:**  
The only external interaction was the inbound retrieval of model artifacts from Hugging Face during setup. No CID was transmitted externally as part of model execution or inference.

## 4. Architecture and Data Boundary
The deployment boundary is as follows:

- **Model source:** DeepSeek OCR artifacts obtained from Hugging Face.
- **Artifact storage:** Extracted model files stored on the Genesis file share.
- **Compute boundary:** Azure VMs and GPU resources provisioned inside Genesis.
- **Serving layer:** vLLM running in Kubernetes pods within the Genesis environment.
- **Network control:** Runtime boxes and serving infrastructure do not have outbound internet access.
- **Data residency:** Input data, OCR processing, and output artifacts remain inside Genesis.
- **CID boundary:** CID is not transferred outside Genesis as part of this solution design.

## 5. Control Assessment

| Control Area | Validation Summary | Evidence / Reference |
|---|---|---|
| **Model provenance** | DeepSeek OCR artifacts were sourced from Hugging Face and stored internally on Genesis. Exact model identifier, version, and checksum should be recorded. | [TBD] |
| **Artifact storage** | Downloaded artifacts were extracted and saved on the Genesis file share. | [TBD] |
| **Internal hosting** | Compute was provisioned within Genesis using Azure VMs, GPUs, and supporting infrastructure. | [TBD] |
| **Serving method** | Model inference is served internally using vLLM in Kubernetes pods. | [TBD] |
| **Data residency** | Input data and OCR outputs remain within Genesis. No CID leaves the Genesis boundary. | [TBD] |
| **Network isolation** | Runtime boxes do not have outbound internet access. | [TBD] |
| **External exposure** | The runtime environment is not internet-enabled. External communication is limited to initial inbound artifact acquisition. | [TBD] |
| **Access control** | Access to storage, compute, and Kubernetes resources should be limited to authorized UBS personnel and service principals. | [TBD] |
| **Operational control** | Model serving is contained within managed internal infrastructure and can be monitored through standard Azure/Kubernetes controls. | [TBD] |
| **Output use** | OCR outputs are generated internally and should remain subject to downstream business validation where material use is involved. | [TBD] |
| **Change management** | Any change to model version, artifact source, vLLM version, infrastructure, or deployment configuration should trigger revalidation. | [TBD] |

## 6. Risk View and Mitigating Position

### 6.1 External Data Leakage Risk
**Risk:** CID or sensitive data could be exposed outside UBS infrastructure.  
**Mitigating position:** All storage, compute, and inference are hosted within Genesis, and runtime boxes have no outbound internet access. CID is not transferred outside Genesis.

### 6.2 Model Artifact Provenance Risk
**Risk:** Untracked or unverified model artifacts may create governance and reproducibility issues.  
**Mitigating position:** Model artifacts were centrally downloaded, extracted, and stored internally. Exact artifact source, version, and checksum should be captured as evidence.

### 6.3 Infrastructure Exposure Risk
**Risk:** Internet-enabled infrastructure could create unnecessary exfiltration or control gaps.  
**Mitigating position:** Serving infrastructure is not internet-accessible for outbound runtime traffic.

### 6.4 Model Output Quality Risk
**Risk:** OCR output may contain errors or incomplete extraction.  
**Mitigating position:** OCR output should be validated through downstream checks, sampling, or human review where the output is used in material workflows.

### 6.5 Change Drift Risk
**Risk:** Later changes to the model, serving stack, or infrastructure may invalidate the current control position.  
**Mitigating position:** Any material change requires review and revalidation.

## 7. Residual Validation Items
Before final sign-off, attach or complete the following:

- exact Hugging Face model name and version,
- model download date,
- artifact checksum or equivalent integrity record,
- Genesis file share path,
- Azure subscription, resource group, and cluster details,
- VM and GPU inventory,
- Kubernetes namespace, deployment, and pod configuration references,
- vLLM version,
- network/security evidence showing no outbound internet access,
- access control evidence for storage, compute, and cluster access,
- sample OCR validation results or business acceptance evidence.

## 8. Validation Conclusion
Based on the current implementation, DeepSeek OCR is hosted and executed fully within the Genesis environment.

The current control position is:

- model artifacts were acquired externally once and then stored internally,
- serving and inference occur entirely within Genesis,
- runtime infrastructure does not have outbound internet access, and
- CID is not transferred outside Genesis as part of this design.

Final approval remains subject to completion of the evidence items listed above and formal sign-off by the relevant business, technical, security, and MRM stakeholders.

## 9. Approval
- **Business owner:** [Name / Date / Decision]
- **Technical owner:** [Name / Date / Decision]
- **Security reviewer:** [Name / Date / Decision]
- **MRM reviewer:** [Name / Date / Decision]

---
**Decision:** Approved / Conditionally Approved / Rejected

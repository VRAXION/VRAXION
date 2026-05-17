# INSTNCT Compliance Boundary

Status: 056 productization planning artifact.

This document defines the first compliance boundary for INSTNCT productization.
It is not legal advice and does not assert compliance with any regulatory
framework.

## Compliance Position

056 does not make INSTNCT compliant for regulated use. It identifies areas that
must be blocked, scoped, or reviewed before pilots or commercial deployment.

## Healthcare Boundary

Healthcare AI can become regulated when used for medical purposes, clinical
decision support, patient-specific recommendations, or safety-critical workflow.

INSTNCT is not ready for:

- Diagnosis.
- Treatment recommendation.
- Triage.
- Medication decision.
- Clinical decision support.
- Emergency prioritization.
- Patient risk scoring.

Allowed planning-only scope:

- Non-clinical administration.
- Internal research support.
- Synthetic training/demo data.
- Visual/audit demonstration.

## Education Boundary

Education AI can become high-risk when it determines access, admission,
learning outcomes, educational level, testing behavior, or other consequential
student outcomes.

INSTNCT is not ready for:

- Grading.
- Admissions.
- Student ranking.
- Placement decisions.
- Discipline decisions.
- Proctoring or prohibited-behavior detection.
- High-stakes profiling.

Allowed planning-only scope:

- Practice generation.
- Explanations.
- Classroom demonstration.
- Teacher support where the teacher remains responsible.

## Privacy Boundary

Before processing regulated or sensitive data, require:

- Data processing agreement.
- Data minimization plan.
- Retention policy.
- Access controls.
- Audit logs.
- Incident response plan.
- De-identification or synthetic data where possible.

## AI Risk Boundary

Before high-risk or consequential use, require:

- Risk classification.
- Intended-use statement.
- Validation plan.
- Human oversight plan.
- Model monitoring.
- Post-deployment incident handling.
- External review where required.

## Source References

- European Commission AI Act overview:
  https://commission.europa.eu/news-and-media/news/ai-act-enters-force-2024-08-01_en
- European Commission health AI overview:
  https://health.ec.europa.eu/ehealth-digital-health-and-care/artificial-intelligence-healthcare_en
- EU AI Act Annex III:
  https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3
- FDA Clinical Decision Support Software guidance:
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- HHS HIPAA covered entities and business associates:
  https://www.hhs.gov/hipaa/for-professionals/covered-entities/index.html
- U.S. Department of Education FERPA overview:
  https://www.ed.gov/about/contact-us/faqs/Student%20Records%20and%20Privacy

## Claim Boundary

056 supports productization planning only. It does not support production
release, public beta promotion, production API readiness, clinical readiness,
school high-stakes readiness, full VRAXION, language grounding, consciousness,
biological/FlyWire equivalence, or physical quantum behavior.


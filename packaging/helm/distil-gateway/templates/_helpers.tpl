{{/* Common naming + labels. */}}

{{- define "distil-gateway.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "distil-gateway.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "distil-gateway.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "distil-gateway.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: distil
{{- end -}}

{{- define "distil-gateway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "distil-gateway.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "distil-gateway.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "distil-gateway.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* Name of the Secret holding the admin token, whether user- or chart-managed. */}}
{{- define "distil-gateway.adminSecretName" -}}
{{- if .Values.adminToken.existingSecret -}}
{{- .Values.adminToken.existingSecret -}}
{{- else -}}
{{- printf "%s-admin" (include "distil-gateway.fullname" .) -}}
{{- end -}}
{{- end -}}

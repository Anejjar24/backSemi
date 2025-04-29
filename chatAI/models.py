from django.db import models

# Create your models here.


class DataAnalysisRequest(models.Model):
    query = models.TextField(help_text="Requête en langage naturel")
    csv_file = models.FileField(upload_to='uploads/', null=True, blank=True)
    generated_code = models.TextField(null=True, blank=True)
    execution_result = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analyse #{self.id}: {self.query[:50]}..."
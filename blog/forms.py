from django import forms

from  .models import BlogEntry

class PostForm(forms.ModelForm):

    class Meta:
        model = BlogEntry
        fields = ('title', 'text',)
from django.shortcuts import render
from django.utils import timezone
from .models import BlogEntry
from .forms import PostForm
from django.shortcuts import render, get_object_or_404
from django.shortcuts import redirect
def listBlogEntries(request):
    posts = BlogEntry.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request, 'blog/post_list.html', {'posts': posts})

def BlogEntryDetails(request, pk):
    BlogEntry.objects.get(pk=pk)

def BlogEntryDetails(request, pk):
    post = get_object_or_404(BlogEntry, pk=pk)
    return render(request, 'blog/post_detail.html', {'post': post})

def createNewBlogEntry(request):
    if request.method == "POST":
        form = PostForm(request.POST)
        if form.is_valid():
            blogEntry = form.save(commit=False)
            blogEntry.author = request.user
            blogEntry.published_date = timezone.now()
            blogEntry.save()
            return redirect('BlogEntryDetails', pk=blogEntry.pk)
    else:
        form = PostForm()
    return render(request, 'blog/post_edit.html', {'form': form})

def editBlogEntry(request, pk):
    blogEntry = get_object_or_404(BlogEntry, pk=pk)
    if request.method == "POST":
        form = PostForm(request.POST, instance=blogEntry)
        if form.is_valid():
            blogEntry = form.save(commit=False)
            blogEntry.author = request.user
            blogEntry.published_date = timezone.now()
            blogEntry.save()
            return redirect('BlogEntryDetails', pk=blogEntry.pk)
    else:
        form = PostForm(request.POST, instance=blogEntry)
        form = PostForm(instance=blogEntry)
    return render(request, 'blog/post_edit.html', {'form': form})



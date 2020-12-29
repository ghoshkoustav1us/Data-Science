# This script shows how to use the client in anonymous mode
# against jira.atlassian.com.
from jira import JIRA
import re
import StringIO

# By default, the client will connect to a JIRA instance started from the Atlassian Plugin SDK
# (see https://developer.atlassian.com/display/DOCS/Installing+the+Atlassian+Plugin+SDK for details).
# Override this with the options parameter.
options = {
    'server': 'https://jira.atlassian.com'}
jira = JIRA(options)

# Get all projects viewable by anonymous users.
projects = jira.projects()

# Sort available project keys, then return the second, third, and fourth keys.
keys = sorted([project.key for project in projects])[2:5]

# Get an issue.
issue = jira.issue('JRA-1330')

# Create an issue.
new_issue = jira.create_issue(project='PROJ_key_or_id', summary='New issue from jira-python',
                              description='Look into this one', issuetype={'name': 'Bug'})


#Create Multiple issue
issue_list = [
{
    'project': {'id': 123},
    'summary': 'First issue of many',
    'description': 'Look into this one',
    'issuetype': {'name': 'Bug'},
},
{
    'project': {'key': 'FOO'},
    'summary': 'Second issue',
    'description': 'Another one',
    'issuetype': {'name': 'Bug'},
},
{
    'project': {'name': 'Bar'},
    'summary': 'Last issue',
    'description': 'Final issue of batch.',
    'issuetype': {'name': 'Bug'},
}]
issues = jira.create_issues(field_list=issue_list)

# requires issue assign permission, which is different from issue editing permission!
jira.assign_issue(issue, 'newassignee')

# Find all comments made by Atlassians on this issue.
atl_comments = [comment for comment in issue.fields.comment.comments
                if re.search(r'@atlassian.com$', comment.author.emailAddress)]

# Add a comment to the issue.
jira.add_comment(issue, 'Comment text')

# Change the issue's summary and description.
issue.update(
    summary="I'm different!", description='Changed the summary to be different.')

# Change the issue without sending updates
issue.update(notify=False, description='Quiet summary update.')

# You can update the entire labels field like this
issue.update(labels=['AAA', 'BBB'])

# Or modify the List of existing labels. The new label is unicode with no
# spaces
issue.fields.labels.append(u'new_text')
issue.update(fields={"labels": issue.fields.labels})

# Send the issue away for good.
issue.delete()

# Linking a remote jira issue (needs applinks to be configured to work)
issue = jira.issue('JRA-1330')
issue2 = jira.issue('XX-23')  # could also be another instance
jira.add_remote_link(issue, issue2)


# upload file from `/some/path/attachment.txt`
jira.add_attachment(issue=issue, attachment='/some/path/attachment.txt')

# read and upload a file :
with open('/some/path/attachment.txt', 'rb') as f:
    jira.add_attachment(issue=issue, attachment=f)


#Get all attachment
for attachment in issue.fields.attachment:
    print("Name: '{filename}', size: {size}".format(
        filename=attachment.filename, size=attachment.size))
    # to read content use `get` method:
    print("Content: '{}'".format(attachment.get()))

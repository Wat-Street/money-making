import fsspec
from pathlib import Path


def extract_components(url: str) -> list:
    """
    expected URL input and examples: 
    - root directory on default (main) branch: https://github.com/Wat-Street/money-making
    - root directory on a specific branch: https://github.com/Wat-Street/money-making/tree/branch_name
    - a folder on a specific branch: https://github.com/Wat-Street/money-making/tree/branch_name/projects/orderbook
    
    from a Github URL, extract:
    - organization (ex. 'Wat-Street')
    - repo (ex.'money-making')
    - ref (the branch, ex. 'main') **optional** default 'main'
    - file path (ex. 'projects/orderbook_test-model') **optional** default root directory
    """
    GH_DOMAIN = 'github.com'
    parts = url.strip().split('/')

    gh_domain = parts[2]
    organization = parts[3]
    repository = parts[4]
    branch = 'main'
    filepath = ''

    if gh_domain.lower() != GH_DOMAIN:
        # make sure this is a github link
        raise Exception('Expected Github domain. Received a domain at: {gh_domain}')
    
    if len(parts) == 5:
        # root directory on default (main) branch
        return organization, repository, branch, filepath
    
    branch = parts[6]

    if len(parts) == 7:
        # root directory on a specific branch
        return organization, repository, branch, filepath
    
    # a folder on a specific branch
    filepath = '/'.join(parts[7:])
    return organization, repository, branch, filepath


def recursive_repo_clone(url: str, destination_folder:str = "temporary-storage"):
    FILESYSTEM_PROTOCOL = "github"

    organization, repository, branch, filepath = extract_components(url)

    destination = Path.cwd() / destination_folder
    destination.mkdir(exist_ok=True, parents=True)
    fs = fsspec.filesystem(FILESYSTEM_PROTOCOL, org=organization, repo=repository, ref=branch)
    fs.get(fs.ls(filepath), destination.as_posix(), recursive=True)


if __name__ == '__main__':
    url = 'https://github.com/Wat-Street/money-making/tree/main/projects/orderbook_test_model'
    url2 = 'https://github.com/Wat-Street/money-making'
    url3 = 'https://github.com/Wat-Street/money-making/tree/harv-extension'
    recursive_repo_clone(url)


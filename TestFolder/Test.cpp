#include <iostream>

using std::cin;
using std::cout;

int main()
{
    int n;
    int a[10001], b[10001], g[10001], k[10001];
    int x, y;
    int target = -1;

    cin >> n;
    for (int i = 0; i < n; i++)
    {
        cin >> a[i] >> b[i] >> g[i] >> k[i];
    }
    cin >> x >> y;

    for (int i = 0; i < n; i++)
    {
        int left = a[i], right = a[i] + g[i], bottom = b[i], top = b[i] + k[i];
        if (x >= left && x <= right && y >= bottom && y <= top)
            target = i + 1;
    }

    cout << target;
    system("pause");

    return 0;
}
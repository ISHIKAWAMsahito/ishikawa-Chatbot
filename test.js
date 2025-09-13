// この汚くて長いコードを貼り付けて、保存時にどう整形されるか試してみてください。

                                                                const userProfile = {
  name: 'Yamada Taro',
  age: 35,
  isAdmin: false,
  hobbies: ['programming', 'reading', 'movies', 'cycling'],
  address: {
    city: 'Tokyo',
    zipCode: '100-0001',
  },
};

function calculateTotalPrice(items, taxRate) {
  let total = 0;
  for (let i = 0; i < items.length; i++) {
    total += items[i].price * items[i].quantity;
  }
  const tax = total * taxRate;
  var finalPrice = total + tax;
  const unusedVariable = 'this should be flagged by eslint';
  console.log('The final price is: ', finalPrice);
  return finalPrice;
}

const shoppingCart = [
  { name: 'Laptop', price: 120000, quantity: 1 },
  { name: 'Mouse', price: 3500, quantity: 1 },
  { name: 'Keyboard', price: 7800, quantity: 2 },
];

calculateTotalPrice(shoppingCart, 0.1);
